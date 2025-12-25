import sys
import socket
import struct
import csv
import time
import random
import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, lfilter, lfilter_zi

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, 
                             QMessageBox, QCheckBox, QComboBox, QGroupBox, QFileDialog,
                             QDialog, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                             QGraphicsRectItem, QGraphicsItem)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QRectF
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter

import pyqtgraph as pg

# --- CONFIGURATION ---
DEFAULT_PORT = 8888
SAMPLE_RATE = 25000.0   # 25 kSPS
WINDOW_SECONDS = 10     
MAX_POINTS = int(SAMPLE_RATE * WINDOW_SECONDS)

# Voltage Reference Math
VOLTAGE_REF = 1.65
ADC_MAX_VAL = 512.0 

# --- FLAPPY BIRD GAME ---
class FlappyBirdGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Flappy Bird")
        self.resize(400, 600)
        
        # Game Constants
        self.gravity = 0.6
        self.jump_force = -10.0
        self.pipe_speed = 5
        self.gap_size = 150
        self.pipe_freq = 1500 # ms
        
        self.scene = QGraphicsScene(0, 0, 400, 600)
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)

        # Game Objects
        self.bird = None
        self.pipes = []
        self.score = 0
        self.score_text = None
        self.game_active = False
        
        # Bird Physics
        self.bird_velocity = 0
        
        # Timers
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_game)
        
        self.pipe_timer = QTimer()
        self.pipe_timer.timeout.connect(self.spawn_pipe)

        self.init_game()

    def init_game(self):
        self.scene.clear()
        self.pipes = []
        self.bird_velocity = 0
        self.score = 0
        self.game_active = False
        
        # Background
        self.scene.setBackgroundBrush(QBrush(QColor("#87CEEB")))
        
        # Bird
        self.bird = self.scene.addEllipse(50, 300, 30, 30, QPen(Qt.PenStyle.NoPen), QBrush(QColor("#FFD700")))
        
        # Score
        self.score_text = self.scene.addText(f"Score: {self.score}")
        self.score_text.setDefaultTextColor(Qt.GlobalColor.white)
        self.score_text.setScale(2)
        self.score_text.setPos(10, 10)
        self.score_text.setZValue(10)

        # Start Message
        self.start_msg = self.scene.addText("Press START or use EMG to Jump")
        self.start_msg.setScale(1.5)
        self.start_msg.setPos(50, 200)

    def start_game(self):
        if self.game_active: return
        
        self.init_game()
        self.start_msg.setVisible(False)
        self.game_active = True
        self.game_timer.start(30) # ~30 FPS
        self.pipe_timer.start(self.pipe_freq)
        self.jump() # Initial jump

    def stop_game(self):
        self.game_active = False
        self.game_timer.stop()
        self.pipe_timer.stop()
        
        game_over = self.scene.addText("GAME OVER")
        game_over.setDefaultTextColor(Qt.GlobalColor.red)
        game_over.setScale(3)
        game_over.setPos(80, 250)
        
        restart = self.scene.addText("Close window or Restart")
        restart.setScale(1.5)
        restart.setPos(100, 320)

    def jump(self):
        if not self.game_active:
            self.start_game()
        else:
            self.bird_velocity = self.jump_force

    def update_game(self):
        # Physics
        self.bird_velocity += self.gravity
        self.bird.moveBy(0, self.bird_velocity)
        
        # Boundaries
        if self.bird.y() > 600 or self.bird.y() < -50:
            self.stop_game()
            return
            
        # Move Pipes
        for pipe_pair in self.pipes:
            top, bottom = pipe_pair
            top.moveBy(-self.pipe_speed, 0)
            bottom.moveBy(-self.pipe_speed, 0)
            
            # Collision
            if self.bird.collidesWithItem(top) or self.bird.collidesWithItem(bottom):
                self.stop_game()
                return
                
            # Score
            if top.x() + 50 < self.bird.x() and top.data(0) != "passed":
                self.score += 1
                self.score_text.setPlainText(f"Score: {self.score}")
                top.setData(0, "passed")
                
        # Remove off-screen pipes
        if self.pipes and self.pipes[0][0].x() < -60:
            top, bottom = self.pipes.pop(0)
            self.scene.removeItem(top)
            self.scene.removeItem(bottom)

    def spawn_pipe(self):
        pipe_height = random.randint(100, 400)
        
        # Top Pipe
        top_pipe = QGraphicsRectItem(400, 0, 50, pipe_height)
        top_pipe.setBrush(QBrush(QColor("#228B22")))
        self.scene.addItem(top_pipe)
        
        # Bottom Pipe
        bottom_y = pipe_height + self.gap_size
        bottom_height = 600 - bottom_y
        bottom_pipe = QGraphicsRectItem(400, bottom_y, 50, bottom_height)
        bottom_pipe.setBrush(QBrush(QColor("#228B22")))
        self.scene.addItem(bottom_pipe)
        
        self.pipes.append((top_pipe, bottom_pipe))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.jump()
        super().keyPressEvent(event)

# --- WORKER THREAD ---
class UDPWorker(QThread):
    data_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = False
        self.sock = None

    def run(self):
        self.running = True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.ip, self.port))
            self.sock.settimeout(1.0)
            
            while self.running:
                try:
                    data, _ = self.sock.recvfrom(4096)
                    
                    # Decode Binary (Signed Short 'h')
                    num_samples = len(data) // 2
                    fmt = '<' + 'h' * num_samples
                    values = struct.unpack(fmt, data)
                    
                    self.data_received.emit(list(values))
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(f"Connection Error: {str(e)}")
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False
        self.wait()

# --- MAIN WINDOW ---
class OscilloscopeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Filtered Scope")
        self.resize(1200, 900)
        
        # --- BUFFERS ---
        self.plot_buffer = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
        self.recording_buffer = [] 
        self.worker = None
        self.game_window = None

        # --- STATE ---
        self.is_recording = False
        self.record_start_time = 0
        
        # --- GAME STATE ---
        self.emg_threshold = 0.5
        self.game_control_active = False
        self.last_jump_time = 0
        self.jump_cooldown = 0.3 # seconds

        # --- SIGNAL GAIN ---
        self.signal_gain = 1.0

        # --- FILTER STATE ---
        self.bp_enabled = False
        self.bp_sos = None   
        self.bp_zi = None    

        self.notch_enabled = False
        self.notch_coeffs = []
        self.notch_zis = []
        self.init_notch_filters()

        # --- UI SETUP ---
        self.init_ui()
        self.apply_theme()
        
        # --- TIMER ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui_loop)
        self.timer.start(33) # 30 FPS

    def init_notch_filters(self):
        freqs = [50.0, 100.0, 150.0]
        quality_factor = 30.0 
        self.notch_coeffs = []
        self.notch_zis = []
        for f in freqs:
            b, a = iirnotch(f, quality_factor, fs=SAMPLE_RATE)
            self.notch_coeffs.append((b, a))
            zi = lfilter_zi(b, a)
            self.notch_zis.append(zi)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- TOP HEADER (Connection + Filters) ---
        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        
        # 1. Connection
        grp_conn = QGroupBox("Connection")
        grp_conn.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #555; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_conn = QHBoxLayout()
        self.txt_ip = QLineEdit("0.0.0.0")
        self.txt_ip.setFixedWidth(80)
        self.txt_port = QLineEdit(str(DEFAULT_PORT))
        self.txt_port.setFixedWidth(50)
        self.btn_start = QPushButton("CONNECT")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_stream)
        layout_conn.addWidget(QLabel("IP:"))
        layout_conn.addWidget(self.txt_ip)
        layout_conn.addWidget(QLabel("Port:"))
        layout_conn.addWidget(self.txt_port)
        layout_conn.addWidget(self.btn_start)
        grp_conn.setLayout(layout_conn)

        # 2. Notch Filter (50Hz Remover)
        grp_notch = QGroupBox("Mains Hum")
        grp_notch.setStyleSheet("QGroupBox { color: #FF9E00; font-weight: bold; border: 1px solid #FF9E00; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_notch = QVBoxLayout()
        
        self.chk_notch = QCheckBox("Remove 50/100/150Hz")
        self.chk_notch.setStyleSheet("color: #FF9E00;")
        self.chk_notch.toggled.connect(self.toggle_notch)
        
        layout_notch.addWidget(self.chk_notch)
        grp_notch.setLayout(layout_notch)

        # 3. Bandpass Filter
        grp_filter = QGroupBox("Bandpass")
        grp_filter.setStyleSheet("QGroupBox { color: #00B4D8; font-weight: bold; border: 1px solid #00B4D8; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_filter = QHBoxLayout()
        
        self.chk_bp = QCheckBox("Enable")
        self.txt_low = QLineEdit("25")
        self.txt_low.setFixedWidth(30)
        self.txt_high = QLineEdit("150")
        self.txt_high.setFixedWidth(30)
        self.combo_order = QComboBox()
        self.combo_order.addItems(["1", "2", "3", "4"])
        self.combo_order.setCurrentIndex(3)
        self.btn_apply = QPushButton("SET")
        self.btn_apply.setObjectName("ApplyButton")
        self.btn_apply.setFixedWidth(40)
        self.btn_apply.clicked.connect(self.recalc_bp_filter)

        layout_filter.addWidget(self.chk_bp)
        layout_filter.addWidget(self.txt_low)
        layout_filter.addWidget(QLabel("-"))
        layout_filter.addWidget(self.txt_high)
        layout_filter.addWidget(QLabel("Hz"))
        layout_filter.addWidget(self.combo_order)
        layout_filter.addWidget(self.btn_apply)
        layout_filter.addWidget(self.btn_apply)
        grp_filter.setLayout(layout_filter)

        # 4. Signal Gain
        grp_gain = QGroupBox("Gain")
        grp_gain.setStyleSheet("QGroupBox { color: #FFCC00; font-weight: bold; border: 1px solid #FFCC00; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_gain = QHBoxLayout()
        
        self.txt_gain = QLineEdit("1.0")
        self.txt_gain.setFixedWidth(40)
        self.txt_gain.editingFinished.connect(self.update_gain)
        
        layout_gain.addWidget(QLabel("x"))
        layout_gain.addWidget(self.txt_gain)
        grp_gain.setLayout(layout_gain)

        # Add all to header
        header_layout.addWidget(grp_conn)
        header_layout.addSpacing(10)
        header_layout.addWidget(grp_notch)
        header_layout.addSpacing(10)
        header_layout.addWidget(grp_notch)
        header_layout.addSpacing(10)
        header_layout.addWidget(grp_filter)
        header_layout.addSpacing(10)
        header_layout.addWidget(grp_gain)
        header_layout.addStretch()

        # --- PLOT AREA (Middle) ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#121212')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.showAxis('bottom', False) 
        self.plot_widget.setLabel('left', 'Voltage (V)', units='V')
        self.plot_widget.setYRange(-1.7, 1.7, padding=0)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='#00B4D8', width=1))
        
        # Threshold Line (Visual feedback)
        self.threshold_line = pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen(color='#FF0', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.threshold_line)
        self.threshold_line.setMovable(True)
        self.threshold_line.sigPositionChangeFinished.connect(self.update_threshold_from_line)

        # --- BOTTOM FOOTER (Recorder + Game) ---
        footer = QFrame()
        footer.setObjectName("Footer")
        footer_layout = QHBoxLayout(footer)
        
        # Recorder Group
        grp_rec = QGroupBox("Data Recorder")
        grp_rec.setStyleSheet("QGroupBox { color: #8F8; font-weight: bold; border: 1px solid #484; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_rec = QHBoxLayout()
        
        self.btn_rec_start = QPushButton("START CAPTURE")
        self.btn_rec_start.setFixedWidth(120)
        self.btn_rec_start.clicked.connect(self.start_recording)
        self.btn_rec_start.setStyleSheet("background-color: #225522; color: #8F8; font-weight: bold;") 
        
        self.btn_rec_stop = QPushButton("STOP CAPTURE")
        self.btn_rec_stop.setFixedWidth(120)
        self.btn_rec_stop.clicked.connect(self.stop_recording)
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_stop.setStyleSheet("background-color: #333; color: #555;") 
        
        self.lbl_rec_timer = QLabel("00:00.000")
        self.lbl_rec_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_rec_timer.setStyleSheet("color: #666; font-family: monospace; font-size: 16px; font-weight: bold;")
        
        layout_rec.addWidget(self.btn_rec_start)
        layout_rec.addWidget(self.btn_rec_stop)
        layout_rec.addSpacing(20)
        layout_rec.addWidget(QLabel("Time:"))
        layout_rec.addWidget(self.lbl_rec_timer)
        layout_rec.addStretch()
        
        grp_rec.setLayout(layout_rec)
        
        # Game Group
        grp_game = QGroupBox("Game Control")
        grp_game.setStyleSheet("QGroupBox { color: #FFD700; font-weight: bold; border: 1px solid #FFD700; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_game = QHBoxLayout()

        layout_game.addWidget(QLabel("Trigger Level:"))
        self.txt_thresh = QLineEdit("0.5")
        self.txt_thresh.setFixedWidth(40)
        self.txt_thresh.editingFinished.connect(self.update_threshold_from_box)
        layout_game.addWidget(self.txt_thresh)
        layout_game.addWidget(QLabel("V"))
        
        self.btn_game = QPushButton("LAUNCH GAME")
        self.btn_game.clicked.connect(self.launch_game)
        self.btn_game.setStyleSheet("background-color: #B8860B; color: white;")
        layout_game.addWidget(self.btn_game)
        
        grp_game.setLayout(layout_game)

        footer_layout.addWidget(grp_rec)
        footer_layout.addSpacing(20)
        footer_layout.addWidget(grp_game)
        
        # --- LAYOUT ASSEMBLY ---
        layout.addWidget(header)
        layout.addWidget(self.plot_widget)
        layout.addWidget(footer)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QFrame#Header { background-color: #2B2B2B; border-bottom: 2px solid #0077B6; padding: 5px; }
            QFrame#Footer { background-color: #222; border-top: 1px solid #444; padding: 10px; }
            QLabel { color: #DDD; font-size: 12px; }
            QLineEdit { background-color: #404040; color: #FFF; border: 1px solid #555; padding: 3px; }
            QComboBox { background-color: #404040; color: #FFF; border: 1px solid #555; }
            QCheckBox { color: #FFF; }
            QPushButton { background-color: #0077B6; color: white; border: none; padding: 5px 10px; border-radius: 3px; font-weight: bold; font-size: 11px; }
            QPushButton#ApplyButton { background-color: #D90429; } 
        """)

    def update_threshold_from_line(self):
        self.emg_threshold = self.threshold_line.value()
        self.txt_thresh.setText(f"{self.emg_threshold:.2f}")

    def update_threshold_from_box(self):
        try:
            val = float(self.txt_thresh.text())
            self.emg_threshold = val
            self.threshold_line.setValue(val)
        except:
            pass

    def update_gain(self):
        try:
            val = float(self.txt_gain.text())
            if val < 0: return # Prevent negative gain if desired, or allow it for inversion
            self.signal_gain = val
        except:
            # Revert to current valid gain if parsing fails
            self.txt_gain.setText(str(self.signal_gain))

    def launch_game(self):
        if self.game_window is None:
            self.game_window = FlappyBirdGame(self)
            self.game_window.finished.connect(self.on_game_closed)
        self.game_window.show()
        self.game_window.raise_()
        self.game_window.activateWindow()
        self.game_control_active = True

    def on_game_closed(self):
        self.game_window = None
        self.game_control_active = False

    def toggle_stream(self):
        if self.btn_start.isChecked():
            try:
                port = int(self.txt_port.text())
            except ValueError:
                self.btn_start.setChecked(False)
                return
            self.btn_start.setText("STOP")
            self.worker = UDPWorker(self.txt_ip.text(), port)
            self.worker.data_received.connect(self.handle_data)
            self.worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Error", e))
            self.worker.start()
        else:
            self.btn_start.setText("CONNECT")
            if self.worker:
                self.worker.stop()
                self.worker = None

    def start_recording(self):
        self.recording_buffer = [] 
        self.record_start_time = time.time()
        self.is_recording = True
        
        self.btn_rec_start.setEnabled(False)
        self.btn_rec_start.setStyleSheet("background-color: #333; color: #555;")
        
        self.btn_rec_stop.setEnabled(True)
        self.btn_rec_stop.setStyleSheet("background-color: #D90429; color: white;") 
        
        self.lbl_rec_timer.setStyleSheet("color: #D90429; font-family: monospace; font-size: 16px; font-weight: bold;")

    def stop_recording(self):
        self.is_recording = False
        
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_start.setStyleSheet("background-color: #225522; color: #8F8; font-weight: bold;")
        
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_stop.setStyleSheet("background-color: #333; color: #555;")
        
        self.lbl_rec_timer.setText("00:00.000")
        self.lbl_rec_timer.setStyleSheet("color: #666; font-family: monospace; font-size: 16px; font-weight: bold;")

        if not self.recording_buffer:
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Capture", "", "CSV Files (*.csv)")
        if filename:
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Sample_Index", "Voltage_V"])
                    for i, val in enumerate(self.recording_buffer):
                        writer.writerow([i, val])
                QMessageBox.information(self, "Saved", f"Saved {len(self.recording_buffer)} samples.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")

    def toggle_notch(self):
        self.notch_enabled = self.chk_notch.isChecked()
        self.init_notch_filters()

    def recalc_bp_filter(self):
        try:
            low = float(self.txt_low.text())
            high = float(self.txt_high.text())
            order = int(self.combo_order.currentText())
            if low >= high: return

            nyquist = SAMPLE_RATE / 2.0
            self.bp_sos = butter(order, [low/nyquist, high/nyquist], btype='bandpass', output='sos')
            self.bp_zi = sosfilt_zi(self.bp_sos)
            self.bp_enabled = self.chk_bp.isChecked()
        except:
            self.bp_enabled = False

    def handle_data(self, values):
        chunk = np.array(values, dtype=float)

        if self.notch_enabled:
            for i in range(len(self.notch_coeffs)):
                b, a = self.notch_coeffs[i]
                chunk, self.notch_zis[i] = lfilter(b, a, chunk, zi=self.notch_zis[i])

        if self.bp_enabled and self.bp_sos is not None:
            chunk, self.bp_zi = sosfilt(self.bp_sos, chunk, zi=self.bp_zi)
            
        voltage_chunk = chunk * (VOLTAGE_REF / ADC_MAX_VAL) * self.signal_gain
        
        self.plot_buffer.extend(voltage_chunk)

        if self.is_recording:
            self.recording_buffer.extend(voltage_chunk)
            
        # --- GAME TRIGGER LOGIC ---
        if self.game_control_active and self.game_window:
            # Check for Peak Activity
            max_val = np.max(np.abs(voltage_chunk))
            
            # Simple Threshold Check
            # Use abs(threshold) to handle negative or positive activation
            trigger_level = abs(self.emg_threshold)
            
            if max_val > trigger_level:
                now = time.time()
                if now - self.last_jump_time > self.jump_cooldown:
                    self.game_window.jump()
                    self.last_jump_time = now

    def update_gui_loop(self):
        if self.plot_buffer:
            self.curve.setData(np.array(self.plot_buffer))

        if self.is_recording:
            elapsed = time.time() - self.record_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            millis = int((elapsed % 1) * 1000)
            self.lbl_rec_timer.setText(f"{minutes:02}:{seconds:02}.{millis:03}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OscilloscopeApp()
    window.show()
    sys.exit(app.exec())
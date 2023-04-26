import os.path
import sys
from pathlib import Path
import yaml
import cv2
import pandas as pd
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, QGraphicsView,
                             QGraphicsScene, QGraphicsEllipseItem, QMainWindow,
                             QGraphicsRectItem, QSizePolicy, QGraphicsPixmapItem, QGraphicsSimpleTextItem,
                             QAction, QMenu, QSystemTrayIcon, QFileDialog, QToolBar)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QTransform, QPixmap, QImage, QIcon, QKeySequence

from setRunParameters import set_run_parameters
from processFrame import process_frame
from qImageProcess import qt_image_process
from plotTrackedPoints import plot_tracked_points
from saveLastFrameNumber import save_last_frame_number
from swapLabels import swap_labels, swap_label_sequences
from propagateFrame import propagate_frame
from updateH5file import update_h5file
from saveFrames import save_frame
from findBadTracking import find_bad_tracking
from moveToIndex import move_to_index


class MovingObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r, k):
        super().__init__(0, 0, r, r)
        self.setPos(x, y)
        if k == 0:
            self.setBrush(Qt.magenta)
        else:
            self.setBrush(Qt.blue)
        self.setAcceptHoverEvents(True)

    # mouse hover event
    def hoverEnterEvent(self, event):
        app.instance().setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event) -> None:
        app.instance().restoreOverrideCursor()

    # mouse click event
    def mousePressEvent(self, event) -> None:
        bpt_loc = [self.pos().x(), self.pos().y()]

        for k1 in gui.body_points_dict.keys():
            for k2 in gui.body_points_dict[k1].keys():
                selected_pts = [gui.body_points_dict[k1][k2][0], gui.body_points_dict[k1][k2][1]]
                if bpt_loc == selected_pts:
                    self.selected_individual = k1
                    self.selected_bodypart = k2

    def mouseMoveEvent(self, event) -> None:
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()

        orig_position = self.scenePos()

        updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(updated_cursor_x, updated_cursor_y))

    def mouseReleaseEvent(self, event) -> None:
        self.new_pos = [self.pos().x(), self.pos().y()]
        gui.body_points_dict[self.selected_individual][self.selected_bodypart] = self.new_pos


class GraphicView(QGraphicsView):
    def __init__(self):
        QGraphicsView.__init__(self)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

    def get_image_size(self):
        img_size = self.image.shape[:2]
        return img_size


class MainGUI(QMainWindow):
    def __init__(self, parent=None, video_name=None, h5_name=None):
        super(MainGUI, self).__init__(parent=parent)
        self.parameters = set_run_parameters()
        self.scale_factor = self.parameters.scale_factor

        # Initialize values
        self.video_name = video_name
        self.h5_name = h5_name
        self.event_use_wasd_keys(use_wasd=False)

        # Load graphics view
        self.view = GraphicView()
        self.view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setCentralWidget(self.view)

        # Getting screen resolution to rescale image
        screen = app.primaryScreen()
        screen_res = screen.size()
        self.screen_width = screen_res.width()
        self.screen_height = screen_res.height()

        # Using Configuration Files ############################################################################
        self.filters = "Any File (*)"
        config_path = Path('.') / 'config.yaml'

        with open(config_path, 'r') as fr:
            config = yaml.load(fr, Loader=yaml.FullLoader)

        self.last_frame_path = Path('.') / 'last_video_frame.yaml'

        if self.last_frame_path.exists():
            with open(self.last_frame_path, 'r') as fr:
                self.last_frame_data = yaml.load(fr, Loader=yaml.FullLoader)

        # File paths ############################################################################
        self.videos_main_path = str(config['videos_main_path'][0])
        self.h5files_main_path = str(config['h5files_path'][0])
        self.save_frame_path = config['frames_path']

        # tray = QSystemTrayIcon()
        # tray.setVisible(True)

        self.frame_number = 0
        self.create_ui()

    def create_ui(self) -> None:
        self.create_action()
        self.create_frame_action()
        self.create_widgets()
        self.create_menu_bar()
        self.create_toolbar()

    def create_menu_bar(self) -> None:
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.open_video_action)
        self.file_menu.addAction(self.open_h5_action)

        # Add this causes the GUI to slow down
        # self.edit_menu = self.menuBar().addMenu("&Edit")
        # self.edit_menu.addAction(self.next_frame_action)
        # self.edit_menu.addAction(self.previous_frame_action)
        # self.edit_menu.addAction(self.jump_forward_action)
        # self.edit_menu.addAction(self.jump_backward_action)
        # self.edit_menu.addAction(self.mark_start_action)
        # self.edit_menu.addAction(self.mark_end_action)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.help_action)

    def create_toolbar(self) -> None:
        self.top_toolbar = QToolBar('Load Video and H5file Toolbar')
        self.addToolBar(self.top_toolbar)
        self.top_toolbar.addAction(self.open_video_action)
        self.top_toolbar.addAction(self.open_h5_action)
        self.top_toolbar.addSeparator()
        self.top_toolbar.addWidget(self.frame_number_widget)

        self.left_side_toolbar = QToolBar('Frame Toolbar')
        self.addToolBar(Qt.LeftToolBarArea, self.left_side_toolbar)
        self.left_side_toolbar.addSeparator()
        self.left_side_toolbar.addWidget(QtWidgets.QLabel('Go To Frame: '))
        self.left_side_toolbar.addWidget(self.goto_frame)
        self.left_side_toolbar.addAction(self.next_frame_action)
        self.left_side_toolbar.addAction(self.previous_frame_action)
        self.left_side_toolbar.addSeparator()
        self.left_side_toolbar.addAction(self.jump_forward_action)
        self.left_side_toolbar.addWidget(self.jump_number)
        self.left_side_toolbar.addAction(self.jump_backward_action)
        self.left_side_toolbar.addWidget(self.swap_labels)

        self.right_side_toolbar = QToolBar('Sequence Toolbar')
        self.addToolBar(Qt.RightToolBarArea, self.right_side_toolbar)
        self.right_side_toolbar.addWidget(QtWidgets.QLabel('Swap Sequence of Frames'))
        self.right_side_toolbar.addAction(self.mark_start_action)
        self.right_side_toolbar.addWidget(self.frame_from)
        self.right_side_toolbar.addAction(self.mark_end_action)
        self.right_side_toolbar.addWidget(self.frame_to)
        self.right_side_toolbar.addWidget(self.swap_sequence_button)
        self.right_side_toolbar.addSeparator()
        self.right_side_toolbar.addWidget(QtWidgets.QLabel('Select Animal'))
        self.right_side_toolbar.addWidget(self.prop_animal)
        self.right_side_toolbar.addWidget(self.prop_forward)
        self.right_side_toolbar.addWidget(self.prop_line)
        self.right_side_toolbar.addWidget(self.prop_backward)
        self.right_side_toolbar.addSeparator()
        self.right_side_toolbar.addWidget(self.done_label_button)
        self.right_side_toolbar.addSeparator()
        self.right_side_toolbar.addWidget(self.save_frame_widget)
        self.right_side_toolbar.addWidget(self.find_bad_tracking_button)
        self.right_side_toolbar.addWidget(self.next_index_button)
        self.right_side_toolbar.addWidget(self.behavior_index_completion)

        self.slider_toolbar = QToolBar('Slider Dock')
        self.addToolBar(Qt.BottomToolBarArea, self.slider_toolbar)
        self.slider_toolbar.addWidget(self.frame_slider_widget)

    def create_action(self) -> None:
        # Open Video
        self.open_video_action = QAction(QIcon.fromTheme("document-open"), " &Load Video File",
                                         self)
        self.open_video_action.setShortcut(QKeySequence.Open)
        self.open_video_action.triggered.connect(self.open_vid_file)

        # Open H5
        self.open_h5_action = QAction(QIcon(), ' &Load H5 File',
                                      self)
        self.open_h5_action.setShortcut(QKeySequence("Ctrl+i"))
        self.open_h5_action.triggered.connect(self.open_h5_file)

        # Help functions
        self.help_action = QAction(QIcon(), '&Show Shortcuts',
                                   self)
        self.help_action.setShortcut(QKeySequence("Ctrl+p"))
        self.help_action.triggered.connect(self.show_shortcuts)

    def create_frame_action(self) -> None:
        self.next_frame_action = QAction(QIcon(), '&Next Frame', self)
        self.next_frame_action.triggered.connect(self.event_next_frame)
        self.next_frame_action.setShortcut(QKeySequence(self.next_frame_key))

        self.previous_frame_action = QAction(QIcon(), 'Previous Frame', self)
        self.previous_frame_action.triggered.connect(self.event_previous_frame)
        self.previous_frame_action.setShortcut(QKeySequence(self.previous_frame_key))

        self.jump_forward_action = QAction(QIcon(), 'Jump Forward', self)
        self.jump_forward_action.triggered.connect(self.event_jump_forward)
        self.jump_forward_action.setShortcut(QKeySequence(self.jump_forward_key))

        self.jump_backward_action = QAction(QIcon(), 'Jump Backward', self)
        self.jump_backward_action.triggered.connect(self.event_jump_backward)
        self.jump_backward_action.setShortcut(QKeySequence(self.jump_backward_key))

        self.mark_start_action = QAction(QIcon(), 'Mark Start', self)
        self.mark_start_action.triggered.connect(self.event_mark_start)
        self.mark_start_action.setShortcut(QKeySequence("Ctrl+,"))

        self.mark_end_action = QAction(QIcon(), 'Mark End', self)
        self.mark_end_action.triggered.connect(self.event_mark_end)
        self.mark_end_action.setShortcut(QKeySequence("Ctrl+."))

    def create_widgets(self) -> None:
        self.frame_number_widget = QtWidgets.QLabel()
        font = self.frame_number_widget.font()
        # font.setPointSize(15)
        self.frame_number_widget.setFont(font)

        self.goto_frame = QtWidgets.QLineEdit()
        self.goto_frame.setPlaceholderText('Enter Frame #')
        # self.goto_frame.setFixedWidth(100)
        self.goto_frame.textChanged.connect(self.event_go_to_frame)
        self.goto_frame.returnPressed.connect(self.event_disable_lineedit)

        self.jump_number = QtWidgets.QLineEdit()
        self.jump_number.setPlaceholderText('Enter steps')
        # self.jump_number.setFixedWidth(100)
        self.jump_number.returnPressed.connect(self.event_disable_lineedit)

        self.swap_labels = QtWidgets.QPushButton('Swap Labels')
        font = self.swap_labels.font()
        # font.setPointSize(10)
        self.swap_labels.setFont(font)
        self.swap_labels.clicked.connect(self.event_swap_frame)
        self.swap_labels.setShortcut(QKeySequence("Ctrl+'"))

        self.frame_from = QtWidgets.QLineEdit()
        self.frame_from.setPlaceholderText('From')
        # self.frame_from.setFixedWidth(120)
        self.frame_from.returnPressed.connect(self.event_disable_lineedit)

        self.frame_to = QtWidgets.QLineEdit()
        self.frame_to.setPlaceholderText('To')
        # self.frame_to.setFixedWidth(120)
        self.frame_to.returnPressed.connect(self.event_disable_lineedit)

        self.swap_sequence_button = QtWidgets.QPushButton('Swap Sequence')
        font = self.swap_sequence_button.font()
        # font.setPointSize(10)
        self.swap_sequence_button.setFont(font)
        self.swap_sequence_button.clicked.connect(self.event_swap_sequence)
        self.swap_sequence_button.setShortcut(QKeySequence("Ctrl+/"))

        self.prop_animal = QtWidgets.QComboBox()

        self.prop_forward = QtWidgets.QPushButton('Propagate Forward')
        self.prop_forward.setFont(font)
        # self.prop_forward.setFixedWidth(150)
        self.prop_forward.clicked.connect(self.event_propagate_forward)
        self.prop_forward.setShortcut(QKeySequence("Ctrl+]"))

        self.prop_line = QtWidgets.QLineEdit()
        self.prop_line.setFont(font)
        # self.prop_line.setFixedWidth(100)
        self.prop_line.setPlaceholderText('Enter steps')
        self.prop_line.returnPressed.connect(self.event_disable_lineedit)

        self.prop_backward = QtWidgets.QPushButton('Propagate Backward')
        self.prop_backward.setFont(font)
        # self.prop_backward.setFixedWidth(150)
        self.prop_backward.clicked.connect(self.event_propagate_backward)
        self.prop_backward.setShortcut(QKeySequence("Ctrl+["))

        self.done_label_button = QtWidgets.QPushButton('Done Relabeling')
        self.done_label_button.setFont(font)
        # self.done_label_button.setFixedWidth(150)
        self.done_label_button.clicked.connect(self.event_done_labeling)
        self.done_label_button.setShortcut(QKeySequence("Ctrl+;"))

        self.save_frame_widget = QtWidgets.QPushButton('Save Frame')
        font = self.save_frame_widget.font()
        # font.setPointSize(10)
        self.save_frame_widget.setFont(font)
        self.save_frame_widget.setShortcut(QKeySequence("Ctrl+s"))
        self.save_frame_widget.clicked.connect(self.event_save_frame)

        self.find_bad_tracking_button = QtWidgets.QPushButton("Find Bad Tracking")
        font = self.save_frame_widget.font()
        # font.setPointSize(10)
        self.find_bad_tracking_button.setFont(font)
        self.find_bad_tracking_button.clicked.connect(self.event_find_bad_tracking)

        self.next_index_button = QtWidgets.QPushButton('Next Bad Tracking')
        self.next_index_button.setFont(font)
        # self.next_index_button.setFixedWidth(120)
        self.next_index_button.clicked.connect(self.event_move_to_index)
        self.next_index_button.setShortcut(QKeySequence("Ctrl+b"))

        self.behavior_index_completion = QtWidgets.QLabel()
        font = self.behavior_index_completion.font()
        # font.setPointSize(8)
        self.behavior_index_completion.setFont(font)

        self.frame_slider_widget = QtWidgets.QSlider(Qt.Horizontal)
        self.frame_slider_widget.setRange(0, 100)
        self.frame_slider_widget.setSingleStep(1)
        self.frame_slider_widget.valueChanged[int].connect(self.event_frame_slider)

    def show_image(self):
        self.gui_height = int(self.image.shape[1] * self.scale_factor * 1.1)
        self.gui_width = int(self.image.shape[0] * self.scale_factor * 1.4)
        self.image = process_frame(self.image, self.screen_height, self.screen_width)
        self.pix = qt_image_process(self.image)
        self.image_graphics = QGraphicsPixmapItem(self.pix)
        self.view.scene.addItem(self.image_graphics)

    def open_vid_file(self) -> None:
        try:
            self.video_name, self.filter_name = QFileDialog.getOpenFileName(self, "Open file",
                                                                            self.videos_main_path,
                                                                            self.filters
                                                                            )
            print(self.video_name)
            self.cap = cv2.VideoCapture(self.video_name)
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            self.indexlength = int(np.ceil(np.log10(self.length)))
            self.frame_slider_widget.setRange(0, self.length)
            ret, self.image = self.cap.read()
            self.show_image()
            self.setGeometry(200, 0, self.gui_width, self.gui_height)
            self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
            if self.last_frame_path.exists():
                if self.video_name in self.last_frame_data.keys():
                    self.move_to_last_labeled_frame()

        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Unable to load the Video \n'
                                                         'Make sure to load right the video')
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'File  does not exist \n'
                                                         'You might need to restart the GUI')
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Expects a video file with a format of avi or mp4')

    def img_plot_tracked_points(self):
        self.body_points_dict = plot_tracked_points(self.h5, self.scale_factor, self.frame_number)
        for k, k1 in enumerate(self.body_points_dict.keys()):
            for k2 in self.body_points_dict[k1].keys():
                x_v = self.body_points_dict[k1][k2][0]
                y_v = self.body_points_dict[k1][k2][1]
                self.moving_object = MovingObject(x_v, y_v, self.parameters.dot_size, k)
                self.moving_object.setToolTip(f'{k1}:{k2}')
                self.view.scene.addItem(self.moving_object)

    def move_to_last_labeled_frame(self) -> None:
        last_frame_output = QtWidgets.QMessageBox.question(self, 'Last Frame',
                                                           'Do you want to go to the last labeled frame',
                                                           buttons=(QtWidgets.QMessageBox.StandardButton.Yes |
                                                                    QtWidgets.QMessageBox.StandardButton.No),
                                                           defaultButton=QtWidgets.QMessageBox.StandardButton.Yes)
        if last_frame_output == QtWidgets.QMessageBox.StandardButton.Yes:
            self.frame_number = self.last_frame_data[self.video_name]
            self.cap.set(1, self.frame_number)
            ret, self.image = self.cap.read()
            self.show_image()
            self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
            self.frame_slider_widget.setValue(self.frame_number)

    def open_h5_file(self) -> None:
        try:
            self.h5_name, self.filter_name = QFileDialog.getOpenFileName(self, "Open file",
                                                                         self.h5files_main_path,
                                                                         "*.h5")
            self.h5 = pd.read_hdf(self.h5_name)
            self.img_plot_tracked_points()

            # Add animals to propagate list
            self.animals_identity = list(self.body_points_dict.keys())
            self.animals_identity.append('both')
            self.prop_animal.addItems(self.animals_identity)
            self.prop_animal.setFixedWidth(100)
            self.prop_animal.setCurrentText(self.animals_identity[-1])

        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    def show_shortcuts(self) -> None:
        QtWidgets.QMessageBox.about(self, "Show Shortcuts",
                                    "Next Frame\t\t --> Right Arrow \n"
                                    "Previous Frame\t --> Left Arrow \n"
                                    "Jump Forward\t --> Up Arrow \n"
                                    "Jump Backward\t --> Down Arrow \n"
                                    "Swap Labels\t --> Ctrl + ' \n"
                                    "Mark Start\t\t --> Ctrl + , \n"
                                    "Mark End\t\t --> Ctrl + . \n"
                                    "Swap Sequence\t --> Ctrl + / \n"
                                    "Propagate Forward\t --> Ctrl + ] \n"
                                    "Propagate Backward\t --> Ctrl + [ \n"
                                    "Relabel\t\t --> Ctrl + l \n"
                                    "Done Labeling\t --> Ctrl + ; \n"
                                    )

    def event_go_to_frame(self) -> None:
        try:
            self.goto_num = self.goto_frame.text()
            if self.goto_num == '':
                self.goto_num = '0'
            try:
                self.frame_number = int(self.goto_num)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered')
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            if self.video_name:
                if self.frame_number > self.length:
                    self.frame_number = self.length
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Frame does not exits')

    # Sliding through the video
    def event_frame_slider(self) -> None:
        try:
            self.frame_number = int(self.frame_slider_widget.value())
            # self.goto_frame.setText(str(self.frame_number))
            if self.video_name:
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()

                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")

        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Unable to read the Video \n'
                                                         'Reload it again')

    # Moving forward through the video one frame at a time.
    def event_next_frame(self) -> None:
        try:
            self.frame_number += 1
            if self.frame_number > self.length:
                self.frame_number = self.length
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            if self.video_name:
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Moving backward through the video one frame at a time.
    def event_previous_frame(self) -> None:
        try:
            self.frame_number -= 1
            if self.frame_number < 0:
                self.frame_number = 0
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            if self.video_name:
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Jump forward a set number of frames
    def event_jump_forward(self) -> None:
        try:
            self.val_num = self.jump_number.text()
            if self.val_num == '':
                self.val_num = '15'
            self.jump_number.setText(str(self.val_num))
            try:
                self.frame_number += int(self.val_num)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered - integer required')
            if self.frame_number > self.length:
                self.frame_number = self.length
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            if self.video_name:
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Jump backward a set number of frames
    def event_jump_backward(self) -> None:
        try:
            self.val_num = self.jump_number.text()
            if self.val_num == '':
                self.val_num = '15'
            self.jump_number.setText(str(self.val_num))
            try:
                self.frame_number -= int(self.val_num)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered - integer required')
            if self.frame_number < 0:
                self.frame_number = 0
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            if self.video_name:
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.show_image()
                    self.img_plot_tracked_points()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Get the frame number to start the sequence swap
    def event_mark_start(self) -> None:
        self.frame_from.setText(str(self.frame_number))

    # Get the frame number to end the sequence swap
    def event_mark_end(self) -> None:
        self.frame_to.setText(str(self.frame_number))

    # Swap the labels for mis-tracked points on the animals for a single frame
    def event_swap_frame(self) -> None:
        try:
            if self.h5_name:
                swap_labels(self.h5, self.frame_number, self.h5_name)
                self.h5 = pd.read_hdf(self.h5_name)
                self.show_image()
                self.img_plot_tracked_points()
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Swap the labels for mis-tracked points on the animals for a sequence of frames.
    # Only works for two tracked animals.
    def event_swap_sequence(self) -> None:
        try:
            if self.h5_name:
                try:
                    self.from_frame_number = int(self.frame_from.text())
                    self.to_frame_number = int(self.frame_to.text())
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered - integer required')
                if self.to_frame_number == self.length:
                    self.to_frame_number = self.to_frame_number
                else:
                    self.to_frame_number += 1
                swap_label_sequences(self.h5, self.from_frame_number, self.to_frame_number, self.h5_name)
                self.h5 = pd.read_hdf(self.h5_name)
                self.show_image()
                self.img_plot_tracked_points()
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Propagate rightly tracked body points from the previous image to the current one
    def event_propagate_forward(self) -> None:
        try:
            if self.h5_name:
                steps = self.prop_line.text()
                if steps == '':
                    steps = '1'
                    self.prop_line.setText(steps)
                try:
                    steps = int(steps)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered - integer required')
                if steps == 1:
                    steps += 1
                animal_ident = self.prop_animal.currentText()
                propagate_frame(self.h5, self.frame_number, self.h5_name, 'forward', steps, animal_ident)
                self.h5 = pd.read_hdf(self.h5_name)
                self.show_image()
                self.img_plot_tracked_points()
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    # Propagate rightly tracked body points from the previous image to the current one
    def event_propagate_backward(self) -> None:
        try:
            if self.h5_name:
                steps = self.prop_line.text()
                if steps == '':
                    steps = '1'
                    self.prop_line.setText(steps)
                try:
                    steps = int(steps)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered - integer required')
                animal_ident = self.prop_animal.currentText()
                propagate_frame(self.h5, self.frame_number, self.h5_name, 'backward', steps, animal_ident)
                self.h5 = pd.read_hdf(self.h5_name)
                self.show_image()
                self.img_plot_tracked_points()
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    def event_done_labeling(self) -> None:
        try:
            self.cap.set(1, self.frame_number)
            ret, self.image = self.cap.read()
            new_points = gui.body_points_dict
            update_h5file(new_points, self.h5, self.frame_number, self.h5_name, self.scale_factor)
            # print(new_points)
            self.h5 = pd.read_hdf(self.h5_name)
            self.show_image()
            self.img_plot_tracked_points()
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Load the Video first')

    def event_save_frame(self) -> None:
        output_path = f'{self.save_frame_path[0]}{Path(self.video_name).stem}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.cap.set(1, self.frame_number)
        ret, image = self.cap.read()
        save_frame(frame=image, index=self.frame_number, indexlength=self.indexlength, output_path=output_path)

    def event_find_bad_tracking(self):
        try:
            find_bad_tracking(self.h5_name)
        except NotImplementedError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Make sure to load the h5 file')

    def event_move_to_index(self) -> None:
        try:
            self.goto_index, self.index_completion = move_to_index(self.h5_name, self.frame_number)
            if self.goto_index == '':
                self.goto_index = '0'
            try:
                self.frame_number = int(self.goto_index)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'ValueError', 'invalid number entered')
            self.goto_frame.setText(str(self.frame_number))
            self.frame_slider_widget.setValue(self.frame_number)
            self.behavior_index_completion.setText(f'Gone through: {self.index_completion}%')
            if self.video_name:
                if self.frame_number > self.length:
                    self.frame_number = self.length
                self.cap.set(1, self.frame_number)
                ret, self.image = self.cap.read()
                if self.h5_name:
                    self.h5 = pd.read_hdf(self.h5_name)
                    self.show_image()
                    self.img_plot_tracked_points()
                else:
                    self.show_image()
                    self.frame_number_widget.setText(f"Frames: {self.frame_number} / {self.length}")
        except AttributeError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Frame does not exits')

    def event_use_wasd_keys(self, use_wasd: bool = True) -> None:
        if use_wasd:
            self.next_frame_key = 'd'
            self.previous_frame_key = 'a'
            self.jump_forward_key = 'w'
            self.jump_backward_key = 's'
        else:
            self.next_frame_key = 'right'
            self.previous_frame_key = 'left'
            self.jump_forward_key = 'up'
            self.jump_backward_key = 'down'

    def event_disable_lineedit(self) -> None:
        self.top_toolbar.setFocus()

    def my_exit_handler(self) -> None:
        try:
            if self.video_name:
                save_last_frame_number(self.frame_number, self.video_name)
        except AttributeError:
            return


app = QApplication(sys.argv)
gui = MainGUI()
gui.setWindowTitle('Pose Correction GUI')
app.aboutToQuit.connect(gui.my_exit_handler)
gui.setGeometry(0, 0, 600, 600)
gui.show()
sys.exit(app.exec())

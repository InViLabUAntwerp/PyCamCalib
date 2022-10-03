# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'calibration_app_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QGridLayout,
    QGroupBox, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSlider, QSpinBox,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QWidget)

from camera_calibration_toolbox.gui.custom_widgets import (DistortionPlotWidget, ReprojErrPlotWidget)
from pyqtgraph import ImageView

class Ui_CalibrationApp(object):
    def setupUi(self, CalibrationApp):
        if not CalibrationApp.objectName():
            CalibrationApp.setObjectName(u"CalibrationApp")
        CalibrationApp.resize(989, 793)
        self.centralwidget = QWidget(CalibrationApp)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QRect(10, 10, 691, 731))
        self.imageTab = QWidget()
        self.imageTab.setObjectName(u"imageTab")
        self.imageTab.setEnabled(True)
        self.imageHorizontalSlider = QSlider(self.imageTab)
        self.imageHorizontalSlider.setObjectName(u"imageHorizontalSlider")
        self.imageHorizontalSlider.setEnabled(False)
        self.imageHorizontalSlider.setGeometry(QRect(70, 510, 601, 22))
        self.imageHorizontalSlider.setMinimum(1)
        self.imageHorizontalSlider.setOrientation(Qt.Horizontal)
        self.imageHorizontalSlider.setTickPosition(QSlider.TicksBelow)
        self.imageHorizontalSlider.setTickInterval(99)
        self.imageSpinBox = QSpinBox(self.imageTab)
        self.imageSpinBox.setObjectName(u"imageSpinBox")
        self.imageSpinBox.setEnabled(False)
        self.imageSpinBox.setGeometry(QRect(10, 510, 42, 22))
        self.imageSpinBox.setMinimum(1)
        self.imageSpinBox.setMaximum(10000)
        self.minImageLabel = QLabel(self.imageTab)
        self.minImageLabel.setObjectName(u"minImageLabel")
        self.minImageLabel.setGeometry(QRect(70, 540, 16, 16))
        self.maxImageLabel = QLabel(self.imageTab)
        self.maxImageLabel.setObjectName(u"maxImageLabel")
        self.maxImageLabel.setGeometry(QRect(650, 540, 31, 20))
        self.reprojErrLabel = QLabel(self.imageTab)
        self.reprojErrLabel.setObjectName(u"reprojErrLabel")
        self.reprojErrLabel.setGeometry(QRect(10, 560, 201, 31))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.reprojErrLabel.setFont(font)
        self.graphicsView = ImageView(self.imageTab)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(10, 10, 661, 481))
        self.tabWidget.addTab(self.imageTab, "")
        self.reprojErrorsTab = QWidget()
        self.reprojErrorsTab.setObjectName(u"reprojErrorsTab")
        self.reprojErrorsTab.setEnabled(True)
        self.reprojErrPlot = ReprojErrPlotWidget(self.reprojErrorsTab)
        self.reprojErrPlot.setObjectName(u"reprojErrPlot")
        self.reprojErrPlot.setGeometry(QRect(10, 10, 661, 481))
        self.RMSreprojErrLabel = QLabel(self.reprojErrorsTab)
        self.RMSreprojErrLabel.setObjectName(u"RMSreprojErrLabel")
        self.RMSreprojErrLabel.setEnabled(True)
        self.RMSreprojErrLabel.setGeometry(QRect(10, 510, 241, 21))
        self.RMSreprojErrLabel.setFont(font)
        self.removeOutliersButton = QPushButton(self.reprojErrorsTab)
        self.removeOutliersButton.setObjectName(u"removeOutliersButton")
        self.removeOutliersButton.setEnabled(False)
        self.removeOutliersButton.setGeometry(QRect(520, 510, 151, 61))
        font1 = QFont()
        font1.setPointSize(10)
        self.removeOutliersButton.setFont(font1)
        self.resetButton = QPushButton(self.reprojErrorsTab)
        self.resetButton.setObjectName(u"resetButton")
        self.resetButton.setEnabled(False)
        self.resetButton.setGeometry(QRect(520, 570, 151, 61))
        self.resetButton.setFont(font1)
        self.tabWidget.addTab(self.reprojErrorsTab, "")
        self.calibrationParamsTab = QWidget()
        self.calibrationParamsTab.setObjectName(u"calibrationParamsTab")
        self.calibrationParamsTab.setEnabled(True)
        self.distortionPlot = DistortionPlotWidget(self.calibrationParamsTab)
        self.distortionPlot.setObjectName(u"distortionPlot")
        self.distortionPlot.setGeometry(QRect(10, 20, 671, 471))
        self.intrinsicsTable = QTableWidget(self.calibrationParamsTab)
        if (self.intrinsicsTable.columnCount() < 2):
            self.intrinsicsTable.setColumnCount(2)
        font2 = QFont()
        font2.setBold(True)
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setFont(font2);
        self.intrinsicsTable.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        __qtablewidgetitem1.setFont(font2);
        self.intrinsicsTable.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.intrinsicsTable.rowCount() < 5):
            self.intrinsicsTable.setRowCount(5)
        __qtablewidgetitem2 = QTableWidgetItem()
        __qtablewidgetitem2.setFont(font2);
        self.intrinsicsTable.setVerticalHeaderItem(0, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        __qtablewidgetitem3.setFont(font2);
        self.intrinsicsTable.setVerticalHeaderItem(1, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        __qtablewidgetitem4.setFont(font2);
        self.intrinsicsTable.setVerticalHeaderItem(2, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        __qtablewidgetitem5.setFont(font2);
        self.intrinsicsTable.setVerticalHeaderItem(3, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        __qtablewidgetitem6.setFont(font2);
        self.intrinsicsTable.setVerticalHeaderItem(4, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        __qtablewidgetitem7.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(0, 0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        __qtablewidgetitem8.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(0, 1, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        __qtablewidgetitem9.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(1, 0, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        __qtablewidgetitem10.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(1, 1, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        __qtablewidgetitem11.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(2, 0, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        __qtablewidgetitem12.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(2, 1, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        __qtablewidgetitem13.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(3, 0, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        __qtablewidgetitem14.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(3, 1, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        __qtablewidgetitem15.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(4, 0, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        __qtablewidgetitem16.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.intrinsicsTable.setItem(4, 1, __qtablewidgetitem16)
        self.intrinsicsTable.setObjectName(u"intrinsicsTable")
        self.intrinsicsTable.setGeometry(QRect(100, 511, 228, 181))
        self.intrinsicsTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.intrinsicsTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.intrinsicsTable.setAutoScroll(False)
        self.intrinsicsTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.intrinsicsTable.setTabKeyNavigation(False)
        self.intrinsicsTable.setProperty("showDropIndicator", False)
        self.intrinsicsTable.setDragDropOverwriteMode(False)
        self.intrinsicsTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.intrinsicsTable.setWordWrap(False)
        self.intrinsicsTable.setCornerButtonEnabled(False)
        self.intrinsicsTable.setRowCount(5)
        self.intrinsicsTable.setColumnCount(2)
        self.intrinsicsTable.horizontalHeader().setMinimumSectionSize(100)
        self.intrinsicsTable.horizontalHeader().setDefaultSectionSize(100)
        self.intrinsicsTable.verticalHeader().setMinimumSectionSize(10)
        self.intrinsicsTable.verticalHeader().setDefaultSectionSize(30)
        self.distortionTable = QTableWidget(self.calibrationParamsTab)
        if (self.distortionTable.columnCount() < 2):
            self.distortionTable.setColumnCount(2)
        __qtablewidgetitem17 = QTableWidgetItem()
        __qtablewidgetitem17.setFont(font2);
        self.distortionTable.setHorizontalHeaderItem(0, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        __qtablewidgetitem18.setFont(font2);
        self.distortionTable.setHorizontalHeaderItem(1, __qtablewidgetitem18)
        if (self.distortionTable.rowCount() < 5):
            self.distortionTable.setRowCount(5)
        __qtablewidgetitem19 = QTableWidgetItem()
        __qtablewidgetitem19.setFont(font2);
        self.distortionTable.setVerticalHeaderItem(0, __qtablewidgetitem19)
        __qtablewidgetitem20 = QTableWidgetItem()
        __qtablewidgetitem20.setFont(font2);
        self.distortionTable.setVerticalHeaderItem(1, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        __qtablewidgetitem21.setFont(font2);
        self.distortionTable.setVerticalHeaderItem(2, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        __qtablewidgetitem22.setFont(font2);
        self.distortionTable.setVerticalHeaderItem(3, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        __qtablewidgetitem23.setFont(font2);
        self.distortionTable.setVerticalHeaderItem(4, __qtablewidgetitem23)
        __qtablewidgetitem24 = QTableWidgetItem()
        __qtablewidgetitem24.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(0, 0, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        __qtablewidgetitem25.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(0, 1, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        __qtablewidgetitem26.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(1, 0, __qtablewidgetitem26)
        __qtablewidgetitem27 = QTableWidgetItem()
        __qtablewidgetitem27.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(1, 1, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        __qtablewidgetitem28.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(2, 0, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        __qtablewidgetitem29.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(2, 1, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        __qtablewidgetitem30.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(3, 0, __qtablewidgetitem30)
        __qtablewidgetitem31 = QTableWidgetItem()
        __qtablewidgetitem31.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(3, 1, __qtablewidgetitem31)
        __qtablewidgetitem32 = QTableWidgetItem()
        __qtablewidgetitem32.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(4, 0, __qtablewidgetitem32)
        __qtablewidgetitem33 = QTableWidgetItem()
        __qtablewidgetitem33.setTextAlignment(Qt.AlignTrailing|Qt.AlignVCenter);
        self.distortionTable.setItem(4, 1, __qtablewidgetitem33)
        self.distortionTable.setObjectName(u"distortionTable")
        self.distortionTable.setGeometry(QRect(450, 510, 229, 182))
        self.distortionTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.distortionTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.distortionTable.setAutoScroll(False)
        self.distortionTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.distortionTable.setTabKeyNavigation(False)
        self.distortionTable.setProperty("showDropIndicator", False)
        self.distortionTable.setDragDropOverwriteMode(False)
        self.distortionTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.distortionTable.setWordWrap(False)
        self.distortionTable.setCornerButtonEnabled(False)
        self.distortionTable.setRowCount(5)
        self.distortionTable.setColumnCount(2)
        self.distortionTable.horizontalHeader().setMinimumSectionSize(100)
        self.distortionTable.horizontalHeader().setDefaultSectionSize(100)
        self.distortionTable.verticalHeader().setMinimumSectionSize(10)
        self.distortionTable.verticalHeader().setDefaultSectionSize(30)
        self.label_3 = QLabel(self.calibrationParamsTab)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(350, 510, 81, 41))
        font3 = QFont()
        font3.setPointSize(8)
        font3.setBold(True)
        self.label_3.setFont(font3)
        self.label_3.setWordWrap(True)
        self.label_5 = QLabel(self.calibrationParamsTab)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 510, 81, 41))
        self.label_5.setFont(font3)
        self.label_5.setWordWrap(True)
        self.tabWidget.addTab(self.calibrationParamsTab, "")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(710, 10, 271, 731))
        self.gridLayoutWidget = QWidget(self.groupBox)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(10, 20, 251, 491))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)

        self.exportButton = QPushButton(self.gridLayoutWidget)
        self.exportButton.setObjectName(u"exportButton")
        self.exportButton.setEnabled(False)

        self.gridLayout.addWidget(self.exportButton, 7, 1, 1, 1)

        self.statusText = QLabel(self.gridLayoutWidget)
        self.statusText.setObjectName(u"statusText")

        self.gridLayout.addWidget(self.statusText, 8, 1, 1, 1)

        self.calibrateButton = QPushButton(self.gridLayoutWidget)
        self.calibrateButton.setObjectName(u"calibrateButton")

        self.gridLayout.addWidget(self.calibrateButton, 6, 1, 1, 1)

        self.tagField = QLineEdit(self.gridLayoutWidget)
        self.tagField.setObjectName(u"tagField")

        self.gridLayout.addWidget(self.tagField, 5, 1, 1, 1)

        self.statusLabel = QLabel(self.gridLayoutWidget)
        self.statusLabel.setObjectName(u"statusLabel")

        self.gridLayout.addWidget(self.statusLabel, 8, 0, 1, 1)

        self.invertCheckBox = QCheckBox(self.gridLayoutWidget)
        self.invertCheckBox.setObjectName(u"invertCheckBox")

        self.gridLayout.addWidget(self.invertCheckBox, 3, 1, 1, 1)

        self.normalizeCheckBox = QCheckBox(self.gridLayoutWidget)
        self.normalizeCheckBox.setObjectName(u"normalizeCheckBox")

        self.gridLayout.addWidget(self.normalizeCheckBox, 2, 1, 1, 1)

        self.namesListWidget = QListWidget(self.gridLayoutWidget)
        self.namesListWidget.setObjectName(u"namesListWidget")

        self.gridLayout.addWidget(self.namesListWidget, 0, 1, 1, 1)

        self.browseButton = QPushButton(self.gridLayoutWidget)
        self.browseButton.setObjectName(u"browseButton")

        self.gridLayout.addWidget(self.browseButton, 1, 1, 1, 1)

        self.label = QLabel(self.gridLayoutWidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.headerField = QLineEdit(self.gridLayoutWidget)
        self.headerField.setObjectName(u"headerField")
        self.headerField.setEnabled(False)

        self.gridLayout.addWidget(self.headerField, 4, 1, 1, 1)

        self.label_2 = QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 4, 0, 1, 1)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 590, 251, 131))
        self.label_6.setPixmap(QPixmap(u"../resources/Logo.jpg"))
        self.label_6.setScaledContents(True)
        CalibrationApp.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(CalibrationApp)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 989, 26))
        CalibrationApp.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(CalibrationApp)
        self.statusbar.setObjectName(u"statusbar")
        CalibrationApp.setStatusBar(self.statusbar)

        self.retranslateUi(CalibrationApp)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(CalibrationApp)
    # setupUi

    def retranslateUi(self, CalibrationApp):
        CalibrationApp.setWindowTitle(QCoreApplication.translate("CalibrationApp", u"InViLab camera calibration app", None))
        self.minImageLabel.setText(QCoreApplication.translate("CalibrationApp", u"1", None))
        self.maxImageLabel.setText(QCoreApplication.translate("CalibrationApp", u"100", None))
        self.reprojErrLabel.setText(QCoreApplication.translate("CalibrationApp", u"Reprojection error = /", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.imageTab), QCoreApplication.translate("CalibrationApp", u"Calibration images", None))
        self.RMSreprojErrLabel.setText(QCoreApplication.translate("CalibrationApp", u"RMS reprojection error = /", None))
        self.removeOutliersButton.setText(QCoreApplication.translate("CalibrationApp", u"Remove outliers", None))
        self.resetButton.setText(QCoreApplication.translate("CalibrationApp", u"Reset", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.reprojErrorsTab), QCoreApplication.translate("CalibrationApp", u"Reproj errors", None))
        ___qtablewidgetitem = self.intrinsicsTable.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("CalibrationApp", u"Value", None));
        ___qtablewidgetitem1 = self.intrinsicsTable.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("CalibrationApp", u"SD", None));
        ___qtablewidgetitem2 = self.intrinsicsTable.verticalHeaderItem(0)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("CalibrationApp", u"fx", None));
        ___qtablewidgetitem3 = self.intrinsicsTable.verticalHeaderItem(1)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("CalibrationApp", u"fy", None));
        ___qtablewidgetitem4 = self.intrinsicsTable.verticalHeaderItem(2)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("CalibrationApp", u"cx", None));
        ___qtablewidgetitem5 = self.intrinsicsTable.verticalHeaderItem(3)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("CalibrationApp", u"cy", None));
        ___qtablewidgetitem6 = self.intrinsicsTable.verticalHeaderItem(4)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("CalibrationApp", u"s", None));

        __sortingEnabled = self.intrinsicsTable.isSortingEnabled()
        self.intrinsicsTable.setSortingEnabled(False)
        ___qtablewidgetitem7 = self.intrinsicsTable.item(0, 0)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem8 = self.intrinsicsTable.item(0, 1)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem9 = self.intrinsicsTable.item(1, 0)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem10 = self.intrinsicsTable.item(1, 1)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem11 = self.intrinsicsTable.item(2, 0)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem12 = self.intrinsicsTable.item(2, 1)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem13 = self.intrinsicsTable.item(3, 0)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem14 = self.intrinsicsTable.item(3, 1)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem15 = self.intrinsicsTable.item(4, 0)
        ___qtablewidgetitem15.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem16 = self.intrinsicsTable.item(4, 1)
        ___qtablewidgetitem16.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        self.intrinsicsTable.setSortingEnabled(__sortingEnabled)

        ___qtablewidgetitem17 = self.distortionTable.horizontalHeaderItem(0)
        ___qtablewidgetitem17.setText(QCoreApplication.translate("CalibrationApp", u"Value", None));
        ___qtablewidgetitem18 = self.distortionTable.horizontalHeaderItem(1)
        ___qtablewidgetitem18.setText(QCoreApplication.translate("CalibrationApp", u"SD", None));
        ___qtablewidgetitem19 = self.distortionTable.verticalHeaderItem(0)
        ___qtablewidgetitem19.setText(QCoreApplication.translate("CalibrationApp", u"k1", None));
        ___qtablewidgetitem20 = self.distortionTable.verticalHeaderItem(1)
        ___qtablewidgetitem20.setText(QCoreApplication.translate("CalibrationApp", u"k2", None));
        ___qtablewidgetitem21 = self.distortionTable.verticalHeaderItem(2)
        ___qtablewidgetitem21.setText(QCoreApplication.translate("CalibrationApp", u"k3", None));
        ___qtablewidgetitem22 = self.distortionTable.verticalHeaderItem(3)
        ___qtablewidgetitem22.setText(QCoreApplication.translate("CalibrationApp", u"p1", None));
        ___qtablewidgetitem23 = self.distortionTable.verticalHeaderItem(4)
        ___qtablewidgetitem23.setText(QCoreApplication.translate("CalibrationApp", u"p2", None));

        __sortingEnabled1 = self.distortionTable.isSortingEnabled()
        self.distortionTable.setSortingEnabled(False)
        ___qtablewidgetitem24 = self.distortionTable.item(0, 0)
        ___qtablewidgetitem24.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem25 = self.distortionTable.item(0, 1)
        ___qtablewidgetitem25.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem26 = self.distortionTable.item(1, 0)
        ___qtablewidgetitem26.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem27 = self.distortionTable.item(1, 1)
        ___qtablewidgetitem27.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem28 = self.distortionTable.item(2, 0)
        ___qtablewidgetitem28.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem29 = self.distortionTable.item(2, 1)
        ___qtablewidgetitem29.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem30 = self.distortionTable.item(3, 0)
        ___qtablewidgetitem30.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem31 = self.distortionTable.item(3, 1)
        ___qtablewidgetitem31.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem32 = self.distortionTable.item(4, 0)
        ___qtablewidgetitem32.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        ___qtablewidgetitem33 = self.distortionTable.item(4, 1)
        ___qtablewidgetitem33.setText(QCoreApplication.translate("CalibrationApp", u"/", None));
        self.distortionTable.setSortingEnabled(__sortingEnabled1)

        self.label_3.setText(QCoreApplication.translate("CalibrationApp", u"Distortion parameters:", None))
        self.label_5.setText(QCoreApplication.translate("CalibrationApp", u"Intrinsic parameters:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calibrationParamsTab), QCoreApplication.translate("CalibrationApp", u"Calibration params", None))
        self.groupBox.setTitle(QCoreApplication.translate("CalibrationApp", u"Calibration options", None))
        self.label_4.setText(QCoreApplication.translate("CalibrationApp", u"Feature tag:", None))
        self.exportButton.setText(QCoreApplication.translate("CalibrationApp", u"Export", None))
        self.statusText.setText(QCoreApplication.translate("CalibrationApp", u"Ready", None))
        self.calibrateButton.setText(QCoreApplication.translate("CalibrationApp", u" Calibrate", None))
        self.statusLabel.setText(QCoreApplication.translate("CalibrationApp", u"Status:", None))
        self.invertCheckBox.setText(QCoreApplication.translate("CalibrationApp", u"Invert", None))
        self.normalizeCheckBox.setText(QCoreApplication.translate("CalibrationApp", u"Normalize", None))
        self.browseButton.setText(QCoreApplication.translate("CalibrationApp", u"Browse", None))
        self.label.setText(QCoreApplication.translate("CalibrationApp", u"File(s):", None))
        self.label_2.setText(QCoreApplication.translate("CalibrationApp", u"Data header:", None))
        self.label_6.setText("")
    # retranslateUi


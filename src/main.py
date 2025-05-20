import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QComboBox, QTabWidget, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QFormLayout, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
import seaborn as sns
from sklearn.decomposition import PCA
import time


class MaterialClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналіз матеріалів для авіації")
        self.setGeometry(100, 100, 1000, 700)
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        # Збільшений шрифт для всіх елементів
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

        # Верхня панель (завантаження даних)
        self.top_panel = QHBoxLayout()

        # Кнопка завантаження даних
        self.btn_load = QPushButton("Завантажити дані")
        self.btn_load.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_load.setStyleSheet("padding: 8px;")
        self.btn_load.clicked.connect(self.load_data)
        self.top_panel.addWidget(self.btn_load)

        # Вибір моделі
        self.model_selector = QComboBox()
        self.model_selector.setFont(QFont("Arial", 10))
        self.model_selector.addItems(["Метод опорних векторів (SVM)", "Випадковий ліс", "Дерево рішень"])
        self.top_panel.addWidget(QLabel("Оберіть модель:"))
        self.top_panel.addWidget(self.model_selector)

        # Велика кнопка навчання моделі
        self.btn_train = QPushButton("НАВЧИТИ МОДЕЛЬ")
        self.btn_train.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_train.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_train.clicked.connect(self.start_training)
        self.top_panel.addWidget(self.btn_train)

        # Індикатор прогресу
        self.progress_label = QLabel("Готово до роботи")
        self.progress_label.setFont(QFont("Arial", 10))
        self.top_panel.addWidget(self.progress_label)

        self.layout.addLayout(self.top_panel)

        # Вкладки для результатів
        self.tabs = QTabWidget()

        # Вкладка з результатами
        self.tab_results = QWidget()
        self.results_layout = QVBoxLayout()

        self.results_group = QGroupBox("Результати класифікації")
        self.results_group.setFont(QFont("Arial", 10, QFont.Bold))
        self.results_group_layout = QVBoxLayout()
        self.results_label = QLabel("Результати будуть тут...")
        self.results_label.setFont(QFont("Arial", 10))
        self.results_label.setStyleSheet("font-size: 12pt;")
        self.results_group_layout.addWidget(self.results_label)
        self.results_group.setLayout(self.results_group_layout)
        self.results_layout.addWidget(self.results_group)

        # Група для тестування
        self.test_group = QGroupBox("Тестування на нових даних")
        self.test_group.setFont(QFont("Arial", 10, QFont.Bold))
        self.test_layout = QFormLayout()

        self.test_table = QTableWidget()
        self.test_table.setFont(QFont("Arial", 10))
        self.test_table.setColumnCount(4)
        self.test_table.setHorizontalHeaderLabels(
            ["Товщина (мм)", "Густина (г/см³)", "Температура (°C)", "Міцність (МПа)"])
        self.test_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.btn_add_row = QPushButton("Додати рядок")
        self.btn_add_row.setFont(QFont("Arial", 10))
        self.btn_add_row.clicked.connect(self.add_test_row)
        self.btn_predict = QPushButton("Прогнозувати")
        self.btn_predict.setFont(QFont("Arial", 10))
        self.btn_predict.clicked.connect(self.predict_test_data)

        self.test_layout.addRow(self.test_table)
        self.test_layout.addRow(self.btn_add_row)
        self.test_layout.addRow(self.btn_predict)

        self.test_result_label = QLabel("")
        self.test_result_label.setFont(QFont("Arial", 10))
        self.test_layout.addRow(self.test_result_label)

        self.test_group.setLayout(self.test_layout)
        self.results_layout.addWidget(self.test_group)

        self.tab_results.setLayout(self.results_layout)

        # Вкладки для візуалізації
        self.init_visualization_tabs()

        self.layout.addWidget(self.tabs)
        self.central_widget.setLayout(self.layout)

        # Додамо кілька тестових рядків за замовчуванням
        self.add_test_row()
        self.add_test_row()

    def init_visualization_tabs(self):
        """Ініціалізація вкладок для візуалізації з пустими графіками"""
        # Вкладка з тепловою картою
        self.tab_heatmap = QWidget()
        self.heatmap_layout = QVBoxLayout()
        self.heatmap_fig, self.heatmap_ax = plt.subplots(figsize=(6, 4))  # Зменшений розмір
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.heatmap_layout.addWidget(self.heatmap_canvas)
        self.tab_heatmap.setLayout(self.heatmap_layout)

        # Вкладка з PCA-графіком
        self.tab_pca = QWidget()
        self.pca_layout = QVBoxLayout()
        self.pca_fig, self.pca_ax = plt.subplots(figsize=(6, 4))
        self.pca_canvas = FigureCanvas(self.pca_fig)
        self.pca_layout.addWidget(self.pca_canvas)
        self.tab_pca.setLayout(self.pca_layout)

        # Вкладка з матрицею плутанини
        self.tab_confusion = QWidget()
        self.confusion_layout = QVBoxLayout()
        self.confusion_fig, self.confusion_ax = plt.subplots(figsize=(6, 4))
        self.confusion_canvas = FigureCanvas(self.confusion_fig)
        self.confusion_layout.addWidget(self.confusion_canvas)
        self.tab_confusion.setLayout(self.confusion_layout)

        self.tabs.addTab(self.tab_results, "Результати")
        self.tabs.addTab(self.tab_heatmap, "Теплова карта")
        self.tabs.addTab(self.tab_pca, "PCA-візуалізація")
        self.tabs.addTab(self.tab_confusion, "Матриця плутанини")

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Виберіть файл даних", "", "CSV (*.csv);;Excel (*.xlsx)")
        if file_path:
            try:
                self.progress_label.setText("Завантаження даних...")
                QApplication.processEvents()  # Оновлюємо інтерфейс

                start_time = time.time()

                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)

                required_columns = ['Товщина (мм)', 'Густина (г/см³)', 'Температура плавлення (°C)',
                                    'Міцність на розрив (МПа)', 'Рекомендація']
                if not all(col in self.df.columns for col in required_columns):
                    raise ValueError("Файл не містить всіх необхідних стовпців")

                self.X = self.df[required_columns[:-1]]
                self.y = self.df["Рекомендація"]

                # Зменшуємо кількість даних для швидкості (якщо дуже багато)
                if len(self.df) > 1000:
                    self.df = self.df.sample(1000, random_state=42)
                    self.X = self.df[required_columns[:-1]]
                    self.y = self.df["Рекомендація"]

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.3, random_state=42
                )

                self.scaler.fit(self.X_train)
                self.X_train = self.scaler.transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)

                elapsed_time = time.time() - start_time
                self.progress_label.setText(f"Дані завантажено ({elapsed_time:.4f} сек)")
                QMessageBox.information(self, "Успіх", f"Дані завантажено та оброблено за {elapsed_time:.4f} секунд")

            except Exception as e:
                self.progress_label.setText("Помилка завантаження")
                QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити дані: {str(e)}")

    def start_training(self):
        if self.df is None:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте дані!")
            return

        self.progress_label.setText("Тренування моделі...")
        self.btn_train.setEnabled(False)
        QApplication.processEvents()  # Оновлюємо інтерфейс

        # Запускаємо тренування в окремому потоці через таймер
        QTimer.singleShot(100, self.train_model)

    def train_model(self):
        try:
            start_time = time.time()

            model_name = self.model_selector.currentText()
            if "Метод опорних векторів" in model_name:
                self.model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
            elif "Випадковий ліс" in model_name:
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            elif "Дерево рішень" in model_name:
                self.model = DecisionTreeClassifier(max_depth=4, random_state=42)

            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)

            # Оцінка моделі
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            report = classification_report(self.y_test, y_pred,
                                           target_names=["Не підходить", "Обмежено підходить", "Ідеально підходить"],
                                           output_dict=True)

            # Форматування звіту класифікації у вигляді таблиці
            report_table = """
            <table border='1' style='border-collapse: collapse; width: 100%;'>
                <tr>
                    <th>Клас</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Підтримка</th>
                </tr>
            """

            for class_name in ["Не підходить", "Обмежено підходить", "Ідеально підходить"]:
                report_table += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{report[class_name]['precision']:.2f}</td>
                    <td>{report[class_name]['recall']:.2f}</td>
                    <td>{report[class_name]['f1-score']:.2f}</td>
                    <td>{report[class_name]['support']}</td>
                </tr>
                """

            report_table += f"""
                <tr style='font-weight: bold;'>
                    <td>Accuracy</td>
                    <td colspan='4'>{report['accuracy']:.2f}</td>
                </tr>
                <tr style='font-weight: bold;'>
                    <td>Macro avg</td>
                    <td>{report['macro avg']['precision']:.2f}</td>
                    <td>{report['macro avg']['recall']:.2f}</td>
                    <td>{report['macro avg']['f1-score']:.2f}</td>
                    <td>{report['macro avg']['support']}</td>
                </tr>
                <tr style='font-weight: bold;'>
                    <td>Weighted avg</td>
                    <td>{report['weighted avg']['precision']:.2f}</td>
                    <td>{report['weighted avg']['recall']:.2f}</td>
                    <td>{report['weighted avg']['f1-score']:.2f}</td>
                    <td>{report['weighted avg']['support']}</td>
                </tr>
            </table>
            """

            # Вивід результатів
            elapsed_time = time.time() - start_time
            result_text = (
                f"<b>Модель:</b> {model_name}<br>"
                f"<b>Час тренування:</b> {elapsed_time:.3f} сек<br><br>"
                f"<b>Основні метрики:</b><br>"
                f"Точність (Accuracy): {accuracy:.2f}<br>"
                f"Precision: {precision:.2f}<br>"
                f"Recall: {recall:.2f}<br>"
                f"F1-Score: {f1:.2f}<br><br>"
                f"<b>Звіт класифікації:</b><br>{report_table}"
            )
            self.results_label.setText(result_text)

            # Оновлення візуалізацій
            self.update_visualizations(y_pred)

            self.progress_label.setText(f"Модель навчена ({elapsed_time:.3f} сек)")

        except Exception as e:
            self.progress_label.setText("Помилка тренування")
            QMessageBox.critical(self, "Помилка", f"Помилка при тренуванні моделі: {str(e)}")
        finally:
            self.btn_train.setEnabled(True)

    def update_visualizations(self, y_pred):
        """Оновлення всіх візуалізацій"""
        try:
            # Очищення графіків перед оновленням
            self.heatmap_ax.clear()
            self.pca_ax.clear()
            self.confusion_ax.clear()

            # Теплова карта (спрощена версія для швидкості)
            self.plot_simplified_heatmap()

            # PCA графік
            self.plot_pca()

            # Матриця плутанини
            self.plot_confusion_matrix(y_pred)

        except Exception as e:
            print(f"Помилка при оновленні візуалізацій: {str(e)}")

    def plot_simplified_heatmap(self):
        """Спрощена версія теплової карти для покращення продуктивності"""
        try:
            # Вибірка даних для прискорення
            plot_df = self.df.sample(min(500, len(self.df)), random_state=42)

            # Групування з усередненням
            heatmap_data = plot_df.groupby([
                pd.cut(plot_df['Температура плавлення (°C)'], bins=15),
                pd.cut(plot_df['Міцність на розрив (МПа)'], bins=15)
            ])['Рекомендація'].mean().unstack()

            # Побудова теплової карти
            sns.heatmap(
                heatmap_data,
                cmap='YlOrRd',
                annot=False,
                ax=self.heatmap_ax,
                vmin=0,
                vmax=2,
                cbar_kws={'label': 'Рекомендація'}
            )

            self.heatmap_ax.set_title("Теплова карта рекомендацій матеріалів")
            self.heatmap_ax.set_xlabel("Міцність на розрив (МПа)", fontsize=9)
            self.heatmap_ax.set_ylabel("Температура плавлення (°C)", fontsize=9)
            self.heatmap_fig.tight_layout()

            self.heatmap_canvas.draw()

        except Exception as e:
            print(f"Помилка при побудові теплової карти: {str(e)}")

    def plot_pca(self):
        """Побудова PCA графіка"""
        try:
            # Використовуємо лише частину даних для швидкості
            sample_size = min(500, len(self.X_train))
            X_sample = self.X_train[:sample_size]
            y_sample = self.y_train[:sample_size]

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_sample)

            scatter = self.pca_ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='viridis')
            self.pca_ax.set_title("PCA-візуалізація даних")
            self.pca_ax.set_xlabel("Перша головна компонента", fontsize=9)
            self.pca_ax.set_ylabel("Друга головна компонента", fontsize=9)
            self.pca_fig.colorbar(scatter, ax=self.pca_ax, label="Рекомендація")
            self.pca_fig.tight_layout()

            self.pca_canvas.draw()

        except Exception as e:
            print(f"Помилка при побудові PCA: {str(e)}")

    def plot_confusion_matrix(self, y_pred):
        """Побудова матриці плутанини"""
        try:
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=["Не підходить (0)", "Обмежено (1)", "Ідеально (2)"])
            disp.plot(ax=self.confusion_ax, cmap='Blues')
            self.confusion_ax.set_title("Матриця плутанини", fontsize=10)
            self.confusion_fig.tight_layout()

            self.confusion_canvas.draw()

        except Exception as e:
            print(f"Помилка при побудові матриці плутанини: {str(e)}")

    def add_test_row(self):
        row = self.test_table.rowCount()
        self.test_table.insertRow(row)
        for col in range(4):
            self.test_table.setItem(row, col, QTableWidgetItem("0.0"))

    def predict_test_data(self):
        if self.model is None:
            QMessageBox.warning(self, "Помилка", "Спочатку навчіть модель!")
            return

        try:
            test_data = []
            for row in range(self.test_table.rowCount()):
                row_data = []
                for col in range(4):
                    item = self.test_table.item(row, col)
                    if item is None or item.text() == "":
                        raise ValueError(f"Порожнє значення в рядку {row + 1}, колонці {col + 1}")
                    row_data.append(float(item.text()))
                test_data.append(row_data)

            test_data_scaled = self.scaler.transform(test_data)
            predictions = self.model.predict(test_data_scaled)
            probabilities = self.model.predict_proba(test_data_scaled)

            result_text = "<b>Результати прогнозу:</b><br><br>"
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pred_text = ["Не підходить (0)", "Обмежено підходить (1)", "Ідеально підходить (2)"][pred]
                result_text += (
                    f"<b>Рядок {i + 1}:</b> Прогноз = {pred_text}<br>"
                    f"Ймовірності: Не підходить={prob[0]:.2f}, Обмежено={prob[1]:.2f}, Ідеально={prob[2]:.2f}<br><br>"
                )

            self.test_result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Помилка при прогнозуванні: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaterialClassifierApp()
    window.show()
    sys.exit(app.exec_())
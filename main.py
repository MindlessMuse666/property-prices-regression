import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')


class PropertyPricesPrediction:
    """
    Класс для прогнозирования цен на недвижимость с использованием линейной и полиномиальной регрессии.
    """

    def __init__(self, data_path):
        """
        Инициализация класса.
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.linear_model = None
        self.polynomial_model = None
        self.poly = None

        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

        self.hist_color = "skyblue"
        self.scatter_color = "royalblue"
        self.linear_color = "tomato"
        self.poly_color = "forestgreen"
        self.data_color = "darkgrey"


    def load_data(self):
        """Загрузка данных из CSV файла."""
        try:
            self.df = pd.read_csv(self.data_path)
            print("Данные успешно загружены.")
        except FileNotFoundError:
            print(f"Ошибка: Файл {self.data_path} не найден.")
            exit()
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            exit()


    def preprocess_data(self):
        """
        Предварительная обработка данных.
        """
        self.df = self.df.dropna()
        print("Удалены строки с пропущенными значениями.\n")


        def extract_avg_size(size_str):
            """
            Извлекает среднее значение площади из строки.
            """
            try:
                if "-" in size_str:
                    sizes = size_str.replace("sqft", "").split("-")
                    avg_size = (float(sizes[0]) + float(sizes[1])) / 2
                    return avg_size
                else:
                    return float(size_str.replace("sqft", ""))
            except:
                return None


        self.df["avg_size"] = self.df["size"].apply(extract_avg_size)
        self.df = self.df.dropna(subset=["avg_size"])
        print('Преобразован столбец "size" в числовой формат (avg_size).\n')

        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")
        self.df = self.df.dropna(subset=["price"])


    def visualize_data(self):
        """
        Визуализация данных.
        """
        # 1. Гистограмма распределения цен
        self._create_histogram("price", "Распределение цен на недвижимость", "Цена", "Количество", "price_distribution.png")

        # 2. Диаграмма рассеяния между площадью и ценой
        self._create_scatter_plot(
            "avg_size", "price", "Зависимость цены от площади", "Площадь (кв. футы)", "Цена", "size_vs_price.png"
        )


    def _create_histogram(self, column, title, xlabel, ylabel, filename):
        """
        Вспомогательная функция для создания гистограммы.
        """
        plt.figure(num=title, figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color=self.hist_color)
        plt.title(title, fontsize=16, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(axis="y", alpha=0.75)
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.show()
        plt.close()


    def _create_scatter_plot(self, x_column, y_column, title, xlabel, ylabel, filename):
        """
        Вспомогательная функция для создания диаграммы рассеяния.
        """
        plt.figure(num=title, figsize=(10, 6))
        sns.scatterplot(x=self.df[x_column], y=self.df[y_column], color=self.scatter_color, alpha=0.7)
        plt.title(title, fontsize=16, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.show()
        plt.close()


    def split_data(self, feature_column="avg_size", target_column="price", test_size=0.2, random_state=42):
        """
        Разделение данных на обучающую и тестовую выборки.
        """
        X = self.df[[feature_column]]
        y = self.df[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print("Данные разделены на обучающую и тестовую выборки.\n")


    def create_linear_model(self):
        """
        Создание и обучение модели линейной регрессии.
        """
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train, self.y_train)
        
        print("Модель линейной регрессии создана и обучена.")


    def create_polynomial_model(self, degree=2):
        """
        Создание и обучение модели полиномиальной регрессии.
        """
        self.poly = PolynomialFeatures(degree=degree)
        X_train_poly = self.poly.fit_transform(self.X_train)
        X_test_poly = self.poly.transform(self.X_test)

        self.polynomial_model = LinearRegression()
        self.polynomial_model.fit(X_train_poly, self.y_train)
        
        print(f"Модель полиномиальной регрессии (степень {degree}) создана и обучена.\n")


    def evaluate_models(self):
        """
        Оценка моделей.
        """
        # Оценка линейной регрессии
        y_pred_linear = self.linear_model.predict(self.X_test)
        mse_linear = mean_squared_error(self.y_test, y_pred_linear)
        r2_linear = r2_score(self.y_test, y_pred_linear)

        print("\nЛинейная регрессия:")
        print(f"MSE: {mse_linear:.2f}")
        print(f"R^2: {r2_linear:.2f}")

        # Оценка полиномиальной регрессии
        X_test_poly = self.poly.transform(self.X_test)
        y_pred_poly = self.polynomial_model.predict(X_test_poly)
        mse_poly = mean_squared_error(self.y_test, y_pred_poly)
        r2_poly = r2_score(self.y_test, y_pred_poly)

        print("\nПолиномиальная регрессия:")
        print(f"MSE: {mse_poly:.2f}")
        print(f"R^2: {r2_poly:.2f}")


    def visualize_results(self):
        """
        Визуализация результатов.
        """
        # Предсказания линейной регрессии
        y_pred_linear = self.linear_model.predict(self.X_test)

        # Предсказания полиномиальной регрессии
        X_test_poly = self.poly.transform(self.X_test)
        y_pred_poly = self.polynomial_model.predict(X_test_poly)

        # Сортировка данных для более красивого отображения линии регрессии
        X_test_sorted = np.sort(self.X_test.values.flatten())
        y_pred_linear_sorted = self.linear_model.predict(X_test_sorted.reshape(-1, 1))

        X_test_poly_values = self.poly.transform(self.X_test)
        y_pred_poly_sorted = self.polynomial_model.predict(self.poly.transform(X_test_sorted.reshape(-1, 1)))

        # Создание графика
        plt.figure(num="Сравнение моделей регрессии", figsize=(12, 8))
        sns.scatterplot(
            x=self.X_test["avg_size"], y=self.y_test, label="Фактические данные", alpha=0.7, color=self.data_color
        )
        plt.plot(X_test_sorted, y_pred_linear_sorted, color=self.linear_color, label="Линейная регрессия", linewidth=2)
        plt.plot(
            X_test_sorted, y_pred_poly_sorted, color=self.poly_color, label="Полиномиальная регрессия", linewidth=2
        )

        plt.title("Сравнение моделей регрессии", fontsize=16, pad=15)
        plt.xlabel("Площадь (кв. футы)", fontsize=12)
        plt.ylabel("Цена", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "model_comparison.png"))
        plt.show()
        plt.close()


    def main(self):
        """
        Запуск процесса прогнозирования цен на недвижимость.
        """
        self.load_data()
        self.preprocess_data()
        self.visualize_data()
        self.split_data()
        self.create_linear_model()
        self.create_polynomial_model()
        self.evaluate_models()
        self.visualize_results()


if __name__ == "__main__":
    data_path = "real_estate_dataset.csv"
    house_price_predictor = PropertyPricesPrediction(data_path)
    house_price_predictor.main()
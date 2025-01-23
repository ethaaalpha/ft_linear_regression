import math, csv

class Analyzer:
    def normalize_list(self, data: list[tuple[float, float]]) -> list[tuple[float, float]]:
        min_x, max_x = min([x for x,_ in data]), max([x for x,_ in data])
        min_y, max_y = min([y for _,y in data]), max([y for _,y in data])
        all_x_normalized = [self.normalize(min_x, max_x, x) for x, _ in data]
        all_y_normalized = [self.normalize(min_y, max_y, y) for _, y in data]

        return list(zip(all_x_normalized, all_y_normalized))

    def denormalize_list(self, original_data: list[tuple[float, float]], normalized_data: list[tuple[float, float]]) -> list[tuple[float, float]]:
        min_x, max_x = min([x for x,_ in original_data]), max([x for x,_ in original_data])
        min_y, max_y = min([y for _,y in original_data]), max([y for _,y in original_data])
        all_x_reverted = [self.denormalize(min_x, max_x, x) for x,_ in normalized_data]
        all_y_reverted = [self.denormalize(min_y, max_y, y) for _,y in normalized_data]
        
        return list(zip(all_x_reverted, all_y_reverted))

    def normalize(self, y_min, y_max, y):
        """Normalization of the value between [0, 1]"""
        return (y - y_min) / (y_max - y_min)

    def denormalize(self, y_min, y_max, y):
        return y * (y_max - y_min) + y_min

    def accuracy(self, y_actual: list[float], y_predicted: list[float]) -> float:
        root_mnse = self.__root_mean_square_error(y_actual, y_predicted)
        range = max(y_actual) - min(y_predicted)
        return 100 - (root_mnse / range) * 100

    def linear_function(self, a: float, x: float, b: float):
        return a * x + b

    def __root_mean_square_error(self, y_actual: list[float], y_predicted: list[float]) -> float:
        errors = [(y_p - y_a)**2 for y_a, y_p in zip(y_actual, y_predicted)]
        return math.sqrt(sum(errors) / len(errors))
    
class CSVManager:
    def load(self, filepath: str) -> list[tuple[float, float]]:
        """Load the CSV file where tuple[0] = x and tuple[1] = y"""
        result: list[tuple[int, int]] = list()

        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))

            if (len(data) > 1):
                for row in data:
                    if (len(row) == 2):
                        try:
                            result.append((float(row[0]), float(row[1])))
                        except ValueError:
                            pass

        return result

    def export(self, filepath: str, *values: tuple[float, float]):
        """Export the values from the training into a specific file"""
        with open(filepath, 'w+') as file:
            for value in values:
                string = f"{value[0]}, {value[1]}"

                file.write(string + "\n")
                print(f"Written in {filepath}: {string}")
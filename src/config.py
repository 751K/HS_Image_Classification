from src.model_init import AVAILABLE_MODELS


class Config:
    def __init__(self):
        self.model_name = self.select_model()
        self.num_epochs = 100
        self.batch_size = 8
        self.num_workers = 4
        self.warmup_steps = 10
        self.learning_rate = 0.001
        self.seed = 42
        self.data_path = '../datasets/Indian/Indian_pines_corrected.mat'
        self.gt_path = '../datasets/Indian/Indian_pines_gt.mat'

    @staticmethod
    def select_model():
        print("Available models:")
        for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
            print(f"{i}. {model_name}")

        while True:
            try:
                choice = int(input("Enter the number of the model you want to use: "))
                if 1 <= choice <= len(AVAILABLE_MODELS):
                    return list(AVAILABLE_MODELS.keys())[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
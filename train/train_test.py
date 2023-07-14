import unittest
from train_new import train

class Test(unittest.TestCase):
    def test_train(self):
        result = train()
        self.assertGreater(result, 90)

if __name__ == "__main__":
    unittest.main()
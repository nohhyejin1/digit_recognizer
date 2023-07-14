import unittest
import os
import random
from infer_new import infer
from parameterized import parameterized
from flask import Flask

app = Flask(__name__)
test_data_path = "../data"

def select_one_rand_jpg(number):
    path = os.path.join(test_data_path, str(number))
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    if jpg_files:
        random_file = random.choice(jpg_files)
        return os.path.join(path, random_file)
    else:
        return None

class TestInfer(unittest.TestCase):
    @parameterized.expand([(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,),])
    def test_infer(self, number):
        _, result = infer(select_one_rand_jpg(number))
        self.assertEqual(result, number)

if __name__ == "__main__":
    success_threshold = 7
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInfer)
    result = unittest.TextTestRunner().run(suite)

    if len(result.errors) > 0:
        print("ERROR occured")
        exit(1)
    
    fail_count = len(result.failures)
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    print(success_count, "/", result.testsRun)

    if success_count >= success_threshold:
        print("SUCCESS")
        exit(0)
    else:
        print("FAILURE")
        exit(1)
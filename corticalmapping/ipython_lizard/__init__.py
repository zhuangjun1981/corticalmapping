import os
PACKAGE_DIR = os.path.dirname(__file__)
TEST_PKL_FOLDER = os.path.join(PACKAGE_DIR,r"res\test_pkls")
TEST_PKLS = [os.path.join(TEST_PKL_FOLDER,pkl_name) for pkl_name in os.listdir(TEST_PKL_FOLDER)]
            
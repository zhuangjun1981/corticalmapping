import os

TEST_PKL_FOLDER = r"\\AIBSDATA2\nc-ophys\JunRetinotopicTestPkls"

if os.path.exists(TEST_PKL_FOLDER):
    TEST_PKLS = [os.path.join(TEST_PKL_FOLDER,pkl_name) for pkl_name in os.listdir(TEST_PKL_FOLDER) if pkl_name.endswith("pkl")]
            
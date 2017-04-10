import unittest
import preprocessor as pp


class MyTestCase(unittest.TestCase):
    def test_parse_filename(self):
        filename='Pointing_Thumb_Up_691_235_50_20170304_110354_649_35_1920_1080.png'
        g = pp.parse_name(filename)

        self.assertEqual('Pointing_Thumb_Up', g['gesture'])
        self.assertEqual('691', g['center_x'])
        self.assertEqual('235', g['center_y'])
        self.assertEqual('50', g['center_r'])
        self.assertEqual('20170304', g['date'])
        self.assertEqual('110354', g['time'])
        self.assertEqual('649', g['min_x'])
        self.assertEqual('35', g['min_y'])
        self.assertEqual('1920', g['max_x'])
        self.assertEqual('1080', g['max_y'])


    def test


if __name__ == '__main__':
    unittest.main()

from src.box_detection import ALL_IDS, PART_A_IDS, PART_B_IDS


def test_17_ids_unique_and_ordered():
    assert len(ALL_IDS) == 17 == len(set(ALL_IDS))
    assert PART_A_IDS == ["1a", "1b", "1c", "1d", "1e"]
    assert PART_B_IDS == [["2A", "2B"], ["3A", "3B"], ["4A", "4B"], ["5A", "5B"], ["6A", "6B"], ["7A", "7B"]]

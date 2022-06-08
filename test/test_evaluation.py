from evaluation import get_topk_rationale


def test_get_topk_rationale():
    attributions = [('[CLS]', 0.0),
                    ('George', -0.02306421836946924),
                    ('Washington', 0.03316721758716733),
                    ('was', -0.0008838851494077826),
                    ('the', 0.019633481145616728),
                    ('first', 0.0001883615294160001),
                    ('president', -0.10349918747141952),
                    ('of', -0.020206038566285882),
                    ('the', 0.11672030854801307),
                    ('United', 0.06971974104061011),
                    ('States', 0.9702335634154521),
                    ('.', 0.16438881073743186),
                    ('[SEP]', 0.0)]
    result = get_topk_rationale(attributions, k=3)
    expected = [10, 11, 8]
    assert result == expected


def test_get_topk_rationale_as_mask():
    attributions = [('[CLS]', 0.0),
                    ('George', -0.02306421836946924),
                    ('Washington', 0.03316721758716733),
                    ('was', -0.0008838851494077826),
                    ('the', 0.019633481145616728),
                    ('first', 0.0001883615294160001),
                    ('president', -0.10349918747141952),
                    ('of', -0.020206038566285882),
                    ('the', 0.11672030854801307),
                    ('United', 0.06971974104061011),
                    ('States', 0.9702335634154521),
                    ('.', 0.16438881073743186),
                    ('[SEP]', 0.0)]
    result = get_topk_rationale(attributions, k=3, return_mask=True)
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    assert result == expected

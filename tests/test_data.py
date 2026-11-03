import numpy as np
import pandas as pd

from stack.data import _parse_afrr, _parse_fcr


def test_parse_fcr_drops_second_tender_without_german_price(tmp_path):
    names = [f"NEGPOS_{h:02d}_{h + 4:02d}" for h in range(0, 24, 4)]
    d = pd.DataFrame({"DATE_FROM": ["2025-10-28"] * 12, "PRODUCTNAME": names * 2,
                      "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]": [40.0, 80, 120, 60, 20, 4] + ["-"] * 6})
    f = tmp_path / "fcr.xlsx"
    d.to_excel(f, index=False)
    out = _parse_fcr(f)
    assert list(out.block) == [0, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(out.fcr_eur_mw_h, [10, 20, 30, 15, 5, 1])   # EUR/MW per 4 h -> per h


def test_parse_afrr_wide(tmp_path):
    prods = [f"{d}_{h:02d}_{h + 4:02d}" for d in ("POS", "NEG") for h in range(0, 24, 4)]
    d = pd.DataFrame({"DATE_FROM": ["2026-01-01"] * 12, "PRODUCT": prods,
                      "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]": np.arange(12.0),
                      "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]": np.arange(12.0) + 1})
    f = tmp_path / "afrr.xlsx"
    d.to_excel(f, index=False)
    out = _parse_afrr(f)
    assert len(out) == 6
    np.testing.assert_allclose(out.afrr_pos_avg, np.arange(6.0))
    np.testing.assert_allclose(out.afrr_neg_marg, np.arange(6.0, 12.0) + 1)

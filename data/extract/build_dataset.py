"""The extraction dataset: what a grandmother actually says, and what a correct reading of it is.

Honest labelling, stated here and in the README: these passages are **authored by the entrant**, in the
register an elderly Vietnamese relative uses when handing the ritual over. They are not recordings. The
corpus recordings (Google FLEURS vi_vn, CC-BY-4.0) are read speech and contain no giỗ instructions, so an
extraction benchmark cannot come from them. Twenty-four authored passages is a small set; it is reported as
small, and every gold quote is asserted to be a literal substring of its passage so the guard's rules can be
tested against it.

Deliberate traps, because they are what the guard exists for:
- passages that name a day without a month ("ngày rằm thì cúng chay" — 15th of which month?),
- passages where the lunar date is given but the *year* is not stated at all,
- passages where a dish is mentioned as something the family does **not** use ("nhà mình không cúng xôi"),
- passages with a relative's opinion but no instruction.

Run: python data/extract/build_dataset.py
"""
from __future__ import annotations

import json
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent

# (id, [pieces], [gold claims]) — pieces are (start_s, end_s, text)
PASSAGES: list[tuple[str, list[tuple[float, float, str]], list[dict]]] = [
    ("ex01", [
        (0.0, 7.0, "Giỗ ông nội là ngày mười hai tháng bảy âm lịch, nhà mình làm buổi trưa."),
        (7.0, 15.0, "Mâm cúng phải có gà luộc và xôi gấc, thêm đĩa chè đậu xanh nữa."),
        (15.0, 22.0, "Con nhớ thắp nhang rồi đợi nhang tàn mới được dọn mâm xuống."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười hai tháng bảy âm lịch", "lunar_day": 12, "lunar_month": 7},
        {"kind": "offering", "quote": "gà luộc và xôi gấc", "lunar_day": None, "lunar_month": None},
        {"kind": "offering", "quote": "đĩa chè đậu xanh", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "đợi nhang tàn mới được dọn mâm xuống", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex02", [
        (0.0, 9.0, "Bà ngoại mất ngày hai mươi tháng chạp, năm nào cũng cúng đúng ngày đó."),
        (9.0, 17.0, "Bà thích bánh tét với dưa món, mình cúng hai thứ đó là đủ."),
        (17.0, 24.0, "Cúng xong thì đốt vàng mã, con nhớ để phần ông Táo trước."),
    ], [
        {"kind": "gio_date", "quote": "ngày hai mươi tháng chạp", "lunar_day": 20, "lunar_month": 12},
        {"kind": "offering", "quote": "bánh tét với dưa món", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "đốt vàng mã", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex03", [
        (0.0, 8.0, "Giỗ của ông Tư là mồng ba tháng ba, năm nay đúng vào ngày thường."),
        (8.0, 16.0, "Nhà mình không cúng xôi, chỉ cúng bánh chưng và canh măng thôi."),
        (16.0, 23.0, "Anh Hai khấn trước, sau đó lần lượt từng người thắp nhang."),
    ], [
        {"kind": "gio_date", "quote": "mồng ba tháng ba", "lunar_day": 3, "lunar_month": 3},
        {"kind": "offering", "quote": "bánh chưng và canh măng", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "Anh Hai khấn trước", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex04", [
        (0.0, 6.0, "Ngày rằm thì cúng chay, còn mùng một thì cúng mặn."),
        (6.0, 14.0, "Giỗ bà Cả là ngày mười lăm tháng tám, làm mâm cỗ mặn bình thường."),
        (14.0, 21.0, "Trên bàn thờ lúc nào cũng phải có trái cây và một ly nước."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười lăm tháng tám", "lunar_day": 15, "lunar_month": 8},
        {"kind": "offering", "quote": "trái cây và một ly nước", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex05", [
        (0.0, 10.0, "Ông ngoại mất năm bảy mươi chín tuổi, ngày hai mươi sáu tháng mười một âm lịch."),
        (10.0, 18.0, "Mỗi năm tới ngày đó, mẹ làm gà hấp muối và nấu nồi cháo trắng."),
        (18.0, 26.0, "Ăn cơm xong mới hoá vàng, không được hoá trước khi cúng xong."),
    ], [
        {"kind": "gio_date", "quote": "ngày hai mươi sáu tháng mười một âm lịch", "lunar_day": 26, "lunar_month": 11},
        {"kind": "offering", "quote": "gà hấp muối và nấu nồi cháo trắng", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "Ăn cơm xong mới hoá vàng", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex06", [
        (0.0, 8.0, "Cô Ba mất ngày mười chín tháng chín, tới đó mình cúng ở chùa."),
        (8.0, 15.0, "Lễ chùa thì chỉ cần hương hoa với mâm ngũ quả."),
        (15.0, 22.0, "Con gái lớn đứng khấn, mấy đứa nhỏ đứng sau."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười chín tháng chín", "lunar_day": 19, "lunar_month": 9},
        {"kind": "offering", "quote": "hương hoa với mâm ngũ quả", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex07", [
        (0.0, 7.0, "Giỗ đầu của chú Năm là ngày mười tháng tư, mời họ hàng hai bên."),
        (7.0, 15.0, "Mâm giỗ đầu phải có thịt kho hột vịt và canh khổ qua, đúng phong tục."),
        (15.0, 23.0, "Mời họ hàng trước ba ngày, đừng để sát ngày mới mời."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười tháng tư", "lunar_day": 10, "lunar_month": 4},
        {"kind": "offering", "quote": "thịt kho hột vịt và canh khổ qua", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "Mời họ hàng trước ba ngày", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex08", [
        (0.0, 9.0, "Bác Hai giỗ ngày mười bốn tháng hai, nhà bác ấy cúng ở quê."),
        (9.0, 16.0, "Mình chỉ cần thắp nhang ở nhà mình, không cần mâm cỗ lớn."),
        (16.0, 24.0, "Một đĩa trái cây với bình trà là đủ, quan trọng là thành tâm."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười bốn tháng hai", "lunar_day": 14, "lunar_month": 2},
        {"kind": "instruction", "quote": "thắp nhang ở nhà mình", "lunar_day": None, "lunar_month": None},
        {"kind": "offering", "quote": "Một đĩa trái cây với bình trà", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex09", [
        (0.0, 8.0, "Giỗ bà nội là ngày rằm tháng bảy, cũng là ngày lễ Vu Lan."),
        (8.0, 15.0, "Năm nào mẹ cũng nấu chè trôi nước và mua một bó hoa cúc vàng."),
        (15.0, 22.0, "Con nhớ cúng chay, bà nội ăn chay cả đời."),
    ], [
        {"kind": "gio_date", "quote": "ngày rằm tháng bảy", "lunar_day": 15, "lunar_month": 7},
        {"kind": "offering", "quote": "chè trôi nước và mua một bó hoa cúc vàng", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "cúng chay", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex10", [
        (0.0, 9.0, "Ông cố mất ngày mười tám tháng giêng, tính theo âm lịch chứ không phải dương lịch."),
        (9.0, 17.0, "Mâm cúng có gà, có xôi, có rượu trắng, không có gì cầu kỳ."),
        (17.0, 25.0, "Con trai trưởng phải có mặt, nếu vắng thì báo trước."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười tám tháng giêng", "lunar_day": 18, "lunar_month": 1},
        {"kind": "offering", "quote": "gà, có xôi, có rượu trắng", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex11", [
        (0.0, 7.0, "Bà mất ngày ba tháng chạp, năm nào cũng nhớ ngày này."),
        (7.0, 15.0, "Cúng bà thì cúng chè, bánh ít và một đĩa xôi vò."),
        (15.0, 22.0, "Đốt nhang xong thì khấn ba lần, xoay mặt về hướng nam."),
    ], [
        {"kind": "gio_date", "quote": "ngày ba tháng chạp", "lunar_day": 3, "lunar_month": 12},
        {"kind": "offering", "quote": "chè, bánh ít và một đĩa xôi vò", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "xoay mặt về hướng nam", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex12", [
        (0.0, 8.0, "Giỗ của anh Ba là hai mươi lăm tháng sáu, làm đơn giản thôi."),
        (8.0, 15.0, "Ảnh hồi sống thích ăn bún chả, nên mình cúng bún chả."),
        (15.0, 21.0, "Nhớ để đũa hai bên chén, đừng để úp chén."),
    ], [
        {"kind": "gio_date", "quote": "hai mươi lăm tháng sáu", "lunar_day": 25, "lunar_month": 6},
        {"kind": "offering", "quote": "cúng bún chả", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "để đũa hai bên chén", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex13", [
        (0.0, 9.0, "Hồi đó tới giờ, giỗ tổ thì cúng ngày mồng mười tháng ba."),
        (9.0, 16.0, "Mâm cỗ phải có một con gà trống và một đĩa xôi gấc."),
        (16.0, 24.0, "Cả họ cùng về, ai không về được thì gửi tiền cúng."),
    ], [
        {"kind": "gio_date", "quote": "ngày mồng mười tháng ba", "lunar_day": 10, "lunar_month": 3},
        {"kind": "offering", "quote": "một con gà trống và một đĩa xôi gấc", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex14", [
        (0.0, 7.0, "Mẹ tôi mất ngày mười một tháng bảy, năm nay lệch sang tháng sau."),
        (7.0, 16.0, "Mỗi lần cúng mẹ, tôi nấu canh chua cá lóc với kho một nồi thịt."),
        (16.0, 23.0, "Cúng xong thì chia cho hàng xóm, đừng để dư trong nhà."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười một tháng bảy", "lunar_day": 11, "lunar_month": 7},
        {"kind": "offering", "quote": "canh chua cá lóc với kho một nồi thịt", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "chia cho hàng xóm", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex15", [
        (0.0, 8.0, "Giỗ bà Hai là ngày hai mươi chín tháng tư, năm nhuận thì làm tháng nhuận."),
        (8.0, 16.0, "Bà thích hoa lay ơn, lần nào cũng mua một bó."),
        (16.0, 23.0, "Thắp nhang xong phải đứng lại một chút rồi mới đi ra."),
    ], [
        {"kind": "gio_date", "quote": "ngày hai mươi chín tháng tư", "lunar_day": 29, "lunar_month": 4},
        {"kind": "offering", "quote": "hoa lay ơn", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex16", [
        (0.0, 8.0, "Bác Năm cúng giỗ vào ba mươi tháng mười, hôm đó cúng sớm hơn mọi năm."),
        (8.0, 16.0, "Mâm cỗ có nem rán, có giò lụa, có dưa hành."),
        (16.0, 22.0, "Con cháu đứng thành hàng, vái ba vái rồi mới tới lượt khấn."),
    ], [
        {"kind": "gio_date", "quote": "ba mươi tháng mười", "lunar_day": 30, "lunar_month": 10},
        {"kind": "offering", "quote": "nem rán, có giò lụa, có dưa hành", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "vái ba vái", "lunar_day": None, "lunar_month": None},
    ]),
    # ------------------------------- held out below this line -------------------------------
    ("ex17", [
        (0.0, 8.0, "Giỗ ông ngoại là ngày mười ba tháng tám, cúng buổi chiều cho mát."),
        (8.0, 16.0, "Ông thích cua rang muối, nên mâm nào cũng có một đĩa."),
        (16.0, 23.0, "Con nhớ hạ mâm trước khi trời tối, đừng để tối mới hạ."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười ba tháng tám", "lunar_day": 13, "lunar_month": 8},
        {"kind": "offering", "quote": "cua rang muối", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "hạ mâm trước khi trời tối", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex18", [
        (0.0, 9.0, "Giỗ bà cố thì tính ngày mười lăm tháng mười hai, sát Tết nên ai cũng bận."),
        (9.0, 17.0, "Bà cố thì cúng đơn giản, một mâm chè với một nải chuối."),
        (17.0, 24.0, "Nếu bận thì cúng trước một ngày, không ai trách."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười lăm tháng mười hai", "lunar_day": 15, "lunar_month": 12},
        {"kind": "offering", "quote": "một mâm chè với một nải chuối", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex19", [
        (0.0, 7.0, "Chú Sáu mất ngày mười sáu tháng năm, tới đó cả nhà về Bến Tre."),
        (7.0, 15.0, "Mâm cúng phải có bánh xèo và cá lóc nướng trui, đúng kiểu miền Nam."),
        (15.0, 22.0, "Thắp nhang xong thì quay phim lại cho mấy đứa ở xa xem."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười sáu tháng năm", "lunar_day": 16, "lunar_month": 5},
        {"kind": "offering", "quote": "bánh xèo và cá lóc nướng trui", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex20", [
        (0.0, 8.0, "Giỗ ông Hai là ngày mười bảy tháng tư, nhớ đúng tháng tư âm lịch."),
        (8.0, 15.0, "Nhà mình kiêng cúng mực, nên đừng mua mực dù ông thích."),
        (15.0, 23.0, "Trước khi cúng phải lau bàn thờ bằng nước sạch, không dùng khăn bếp."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười bảy tháng tư", "lunar_day": 17, "lunar_month": 4},
        {"kind": "instruction", "quote": "lau bàn thờ bằng nước sạch", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex21", [
        (0.0, 9.0, "Em út mất trẻ lắm, giỗ ngày mười tháng hai, cũng gọi là ngày giỗ non."),
        (9.0, 16.0, "Mẹ nấu cháo gà với mua một ít bánh kẹo, con nít thích gì cúng đó."),
        (16.0, 22.0, "Không đốt vàng mã nhiều, chỉ một ít thôi."),
    ], [
        {"kind": "gio_date", "quote": "giỗ ngày mười tháng hai", "lunar_day": 10, "lunar_month": 2},
        {"kind": "offering", "quote": "cháo gà với mua một ít bánh kẹo", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "không đốt vàng mã nhiều", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex22", [
        (0.0, 8.0, "Bà nội mất ngày mồng năm tháng chín, năm đó trời mưa rất lớn."),
        (8.0, 16.0, "Cúng bà thì cúng xôi đậu xanh, thêm nồi gà kho gừng."),
        (16.0, 23.0, "Con dâu trưởng dọn mâm, con gái thắp nhang."),
    ], [
        {"kind": "gio_date", "quote": "ngày mồng năm tháng chín", "lunar_day": 5, "lunar_month": 9},
        {"kind": "offering", "quote": "xôi đậu xanh, thêm nồi gà kho gừng", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "Con dâu trưởng dọn mâm", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex23", [
        (0.0, 7.0, "Giỗ cụ nội là ngày hai mươi tháng bảy, cúng ở từ đường."),
        (7.0, 15.0, "Mâm cỗ phải có một mâm ngũ quả và một con gà luộc nguyên con."),
        (15.0, 22.0, "Trưởng tộc khấn trước, sau đó mới tới các chi."),
    ], [
        {"kind": "gio_date", "quote": "ngày hai mươi tháng bảy", "lunar_day": 20, "lunar_month": 7},
        {"kind": "offering", "quote": "một mâm ngũ quả và một con gà luộc nguyên con", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "Trưởng tộc khấn trước", "lunar_day": None, "lunar_month": None},
    ]),
    ("ex24", [
        (0.0, 8.0, "Dì Bảy mất ngày mười hai tháng mười, tới ngày đó dì hay về nhà mình."),
        (8.0, 16.0, "Cúng dì thì cúng bánh bèo và chè bắp, dì gốc Quảng Nam."),
        (16.0, 23.0, "Đừng quên đốt đèn trên bàn thờ suốt đêm hôm đó."),
    ], [
        {"kind": "gio_date", "quote": "ngày mười hai tháng mười", "lunar_day": 12, "lunar_month": 10},
        {"kind": "offering", "quote": "bánh bèo và chè bắp", "lunar_day": None, "lunar_month": None},
        {"kind": "instruction", "quote": "đốt đèn trên bàn thờ suốt đêm", "lunar_day": None, "lunar_month": None},
    ]),
]

HELDOUT_IDS = {f"ex{i:02d}" for i in range(17, 25)}


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT.parent.parent))
    from core import textnorm

    train, heldout = [], []
    problems = []
    for pid, pieces, gold in PASSAGES:
        transcript = " ".join(p[2] for p in pieces)
        for claim in gold:
            if not textnorm.contained(claim["quote"], transcript):
                problems.append(f"{pid}: gold quote not a substring: {claim['quote']!r}")
        record = {"id": pid, "transcript": transcript,
                  "pieces": [[s, e, t] for s, e, t in pieces], "gold": gold}
        (heldout if pid in HELDOUT_IDS else train).append(record)

    for name, rows in (("train", train), ("heldout", heldout)):
        path = ROOT / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT.parent.parent)}: {len(rows)} records")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nevery gold quote is a literal substring of its passage: OK")
    print(textwrap.dedent(f"""
        passages: {len(PASSAGES)}  train: {len(train)}  held-out: {len(heldout)}
        gold claims: {sum(len(g) for _, _, g in PASSAGES)}
        authored by the entrant — stated as such in the README and in the post.
    """))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

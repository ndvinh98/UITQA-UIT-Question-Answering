from reader import QA
model = QA('path')


doc = "anh Vinh sinh năm 1998, sở hữu khối tài sản 350 tỷ USD. Quê nhà ở Tây Nguyên"

q = 'ai sinh năm 1998 ?'

answers = model.getPredictions(q,doc)

print(answers)

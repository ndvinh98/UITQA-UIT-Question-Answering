from reader import QA
model = QA('/content/data/trainning')


doc = "anh Vinh sinh năm 1998, sở hữu khối tài sản 350 tỷ USD. Quê nhà ở Tây Nguyên"

q = 'ai sinh năm 1998 ?'

answer = model.predict(doc,q)

print(answer['answer'])
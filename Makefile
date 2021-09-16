
main:
	python main.py

preprocessing:
	./scripts/preprocessing.sh

similarity:
	./scripts/similarity.sh

download:
	# wget --http-user=$(USER) --http-passwd=$(PASSWORD) https://www-lsm.naist.jp/~t-hara/data/similarity_measures.tar.gz
	tar zxvf similarity_measures.tar.gz
	# wget https://www.ieice.org/~rising/AI-5G/dataset/theme2-NEC/dataset_and_issue.tar.gz
	# tar zxvf dataset_and_issue.tar.gz

unzip:
	tar zxvf similarity_measures.tar.gz


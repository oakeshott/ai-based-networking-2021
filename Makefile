train:
	./scripts/train.sh

test:
	./scripts/test.sh

preprocessing:
	./scripts/preprocessing.sh

similarity:
	./scripts/similarity.sh

download:
	wget --http-user=$(USER) --http-passwd=$(PASSWORD) https://www.ieice.org/cs/rising/jpn/2021/itu-nec/dataset_and_issue.tar.gz
	tar zxvf dataset_and_issue.tar.gz

probe:
	./scripts/probe_info.sh

docker:
	docker-compose up -d

decompress:
	tar zxvf similarity_measures.tar.gz

clean:
	./scripts/kill-docker.sh


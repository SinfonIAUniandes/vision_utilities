start-vision_utilities:
	docker build --network=host -t sinfonia:vision_utilities -f ./docker/Dockerfile ..
	docker run -it --rm sinfonia:vision_utilities
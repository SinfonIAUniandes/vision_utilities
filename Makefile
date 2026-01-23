ros_ip := 157.253.113.200
ros_master_ip := 157.253.113.233

build_vision_utilities_complete:
	docker build --ssh default --network=host -t sinfonia_vision_utilities:complete -f ./docker/Dockerfile .
	docker build --ssh default --network=host -t sinfonia_vision_utilities:basic -f ./docker/Dockerfile.basic .
build_vision_utilities_basic:
	docker build --ssh default --network=host -t sinfonia_vision_utilities:basic -f ./docker/Dockerfile.basic .
run_vision_utilities_complete:
	docker run -e ROS_IP=$(ros_ip) -e ROS_MASTER_URI=http://$(ros_master_ip):11311 -it --rm --network=host sinfonia_vision_utilities:complete
run_vision_utilities_basic:
	docker run -e ROS_IP=$(ros_ip) -e ROS_MASTER_URI=http://$(ros_master_ip):11311 -it --rm --network=host sinfonia_vision_utilities:basic
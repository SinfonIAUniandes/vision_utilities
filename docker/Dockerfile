# syntax=docker/dockerfile:1.4
FROM robotics:ros1-runtime

WORKDIR ${ROS_WORKSPACE}/src

RUN --mount=type=ssh \
    mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git clone git@github.com:SinfonIAUniandes/perception_msgs.git ${ROS_WORKSPACE}/src/perception_msgs/ && \
    git clone git@github.com:SinfonIAUniandes/robot_toolkit_msgs.git ${ROS_WORKSPACE}/src/robot_toolkit_msgs/

RUN apt-get update && apt-get install -y pkg-config libcairo2-dev libgl1 ros-noetic-cv-bridge

COPY . ${ROS_WORKSPACE}/src/vision_utilities/

WORKDIR ${ROS_WORKSPACE}/src/vision_utilities

ENV VU_AI_ENABLED=true

RUN pip install -r requirements_basic.txt
RUN pip install -r requirements_ai.txt

WORKDIR ${ROS_WORKSPACE}
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["rosrun", "vision_utilities", "vision_utilities.py", "--with_pepper", "--start_cameras"]

FROM robotics:ros1-runtime

WORKDIR ${ROS_WORKSPACE}/src

COPY ./vision_utilities/ ${ROS_WORKSPACE}/src/vision_utilities/
COPY ./perception_msgs/ ${ROS_WORKSPACE}/src/perception_msgs/
COPY ./robot_toolkit_msgs/ ${ROS_WORKSPACE}/src/robot_toolkit_msgs/

WORKDIR ${ROS_WORKSPACE}/src/vision_utilities

RUN pip install -r all_requirements.txt

WORKDIR ${ROS_WORKSPACE}
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["rosrun", "vision_utilities", "vision_utilities.py"]

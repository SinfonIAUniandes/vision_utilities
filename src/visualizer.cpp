#include <ros/ros.h>
#include <ros/master.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <perception_msgs/Polygon.h>
#include <perception_msgs/visualize_polygon_topic_srv.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <boost/bind.hpp>

// constants in the same file
namespace constants {
    const std::string POLYGON_RENDERING_NAME = "vision_utilities_polygon_rendering";
    const std::string SERVICE_RENDER_POLYGON_TOPIC = "/vision_utilities/rendering/visualize_polygon_topic_srv";
    const std::string PEPPER_FRONT_CAMERA = "/robot_toolkit_node/camera/front/image_raw";
    const std::string TOPIC_POLYGON_RENDERER = "/vision_utilities/rendering/polygons";
}

class PolygonRenderer {
public:
    PolygonRenderer(ros::NodeHandle& nh) : nh_(nh) {
        service_ = nh_.advertiseService(constants::SERVICE_RENDER_POLYGON_TOPIC,
                                        &PolygonRenderer::serviceCallback, this);
        cam_sub_ = nh_.subscribe(constants::PEPPER_FRONT_CAMERA, 10,
                                 &PolygonRenderer::cameraCallback, this);
        pub_ = nh_.advertise<sensor_msgs::Image>(constants::TOPIC_POLYGON_RENDERER, 10);
    }

private:
    struct TopicInfo {
        ros::Subscriber sub;
        std::vector<double> last;
        bool has_last = false;
    };

    bool serviceCallback(perception_msgs::visualize_polygon_topic_srv::Request &req,
                         perception_msgs::visualize_polygon_topic_srv::Response &res)
    {
        ros::master::V_TopicInfo master_topics;
        if (!ros::master::getTopics(master_topics)) {
            res.visualizing = false;
            return true;
        }

        bool found = false;
        for (const auto &t : master_topics) {
            if (t.datatype == "perception_msgs/Polygon" && t.name == req.polygon_topic_name) {
                found = true;
                break;
            }
        }

        if (!found) {
            res.visualizing = false;
            return true;
        }

        std::lock_guard<std::mutex> lock(topics_mutex_);
        ros::Subscriber sub = nh_.subscribe<perception_msgs::Polygon>(
            req.polygon_topic_name, 10,
            boost::bind(&PolygonRenderer::polygonCallback, this, _1, req.polygon_topic_name));

        TopicInfo info;
        info.sub = sub;
        info.has_last = false;
        info.last.clear();
        topics_map_[req.polygon_topic_name] = info;

        res.visualizing = true;
        return true;
    }

    void polygonCallback(const perception_msgs::Polygon::ConstPtr &msg, const std::string &topic) {
        std::lock_guard<std::mutex> lock(topics_mutex_);
        TopicInfo &info = topics_map_[topic];
        info.last.clear();
        info.last.reserve(msg->polygon.size());
        for (auto v : msg->polygon) info.last.push_back(static_cast<double>(v));
        info.has_last = !info.last.empty();
    }

    void cameraCallback(const sensor_msgs::ImageConstPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat image = cv_ptr->image;

        std::lock_guard<std::mutex> lock(topics_mutex_);
        for (auto &kv : topics_map_) {
            const TopicInfo &info = kv.second;
            if (!info.has_last || info.last.empty()) continue;

            std::vector<double> polygons = info.last;
            const int w = image.cols;
            const int h = image.rows;

            for (size_t i = 0; i < polygons.size(); ++i) {
                if ((i % 2) == 0)
                    polygons[i] = std::round(polygons[i] * static_cast<double>(w));
                else
                    polygons[i] = std::round(polygons[i] * static_cast<double>(h));
            }

            for (size_t i = 0; i + 3 < polygons.size(); i += 4) {
                int x1 = static_cast<int>(polygons[i]);
                int y1 = static_cast<int>(polygons[i + 1]);
                int x2 = static_cast<int>(polygons[i + 2]);
                int y2 = static_cast<int>(polygons[i + 3]);
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            }
        }

        sensor_msgs::ImagePtr out_msg =
            cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, image).toImageMsg();
        pub_.publish(out_msg);
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber cam_sub_;
    ros::ServiceServer service_;
    std::unordered_map<std::string, TopicInfo> topics_map_;
    std::mutex topics_mutex_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, constants::POLYGON_RENDERING_NAME, ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    PolygonRenderer node(nh);
    ros::spin();
    return 0;
}

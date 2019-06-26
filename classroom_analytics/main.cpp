// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <chrono>  // NOLINT
#include <gflags/gflags.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <ext_list.hpp>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include <ie_iextension.h>
#include <cstring>
#include "action_detector.hpp"
#include "cnn.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "image_grabber.hpp"
#include "logger.hpp"
#include <iostream>
#include <thread>
#include <queue>
#include <atomic>
#include <csignal>
#include <mutex>
#include <syslog.h>
#include <fstream>
// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#include"../include/classroom_analytics.hpp"

#include <ctime>

using namespace std;
using namespace cv;
using namespace dnn;

double totalEmotions = 0;
double happinessEmotions = 0;
// OpenCV related variables
Mat frame, blob, sentBlob, poseBlob;
int delay = 5;
Net net, sentnet, posenet;
bool sentChecked = false;
bool poseChecked = false;

int backendId;
int targetId;
float confidenceFace;
float confidenceMood;

string subject;
char checkTime[20];	

// flag to control background threads
atomic<bool> keepRunning(true);

// flag to handle UNIX signals
static volatile sig_atomic_t sig_caught = 0;

// currentInfo contains the latest ClassroomInfo tracked by the application.
ClassroomInfo currentInfo;
queue<Mat> nextImage;
String currentPerf;

mutex m, m1, m2;

// nextImageAvailable returns the next image from the queue in a thread-safe way
Mat nextImageAvailable();
// addImage adds an image to the queue in a thread-safe way
ClassroomInfo getCurrentInfo();

Mat nextImageAvailable() {
	Mat rtn;
	m.lock();
	if (!nextImage.empty()) {
		rtn = nextImage.front();
		nextImage.pop();
	}
	m.unlock();
	return rtn;
}

// addImage adds an image to the queue in a thread-safe way
void addImage(Mat img) {
	m.lock();
	if (nextImage.empty()) {
		nextImage.push(img);
	}
	m.unlock();
}

std::vector<std::string> getStudentNames(std::string path) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(path, pt);
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt.end();
    std::vector<std::string> students;
    for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
       students.push_back(it->first);
    }
   return students;
}
// getCurrentInfo returns the most-recent ClassroomInfo for the application.
ClassroomInfo getCurrentInfo() {
	ClassroomInfo rtn;
	m2.lock();
	rtn = currentInfo;
	m2.unlock();
	return rtn;
}

void updateInfo(ClassroomInfo info) {
	m2.lock();
	if (currentInfo.students < info.students) {
		currentInfo.students = info.students;
	}

	if (currentInfo.lookers < info.lookers) {
		currentInfo.lookers = info.lookers;
	}
	for (pair<Sentiment, int> element : info.sent) {
		Sentiment s = element.first;
		if (currentInfo.sent[s] < info.sent[s]) {
			currentInfo.sent[s] = info.sent[s];
		}
	}
	m2.unlock();
}

//Classroom happiness index
double happinessCal(double totalEmotions,double happinessEmotions) {
	double index;
	index =  (happinessEmotions / totalEmotions) * 100;
	return index;
}

//Class participation Index for sitting
double participationCal(double standingStudents, double totalStudents){
	double index;
	index =  (standingStudents / totalStudents) * 100;
	return index;
}

//Classroom attentive index
double attentiveCal(double totalStudents, double attentiveStudents) {
	double index;
	index =  (attentiveStudents / totalStudents) * 100;
	return index;
}

// Function called by worker thread to process the next available video frame.
void frameRunner() {
	while (keepRunning.load()) {
		Mat next = nextImageAvailable();
		if (!next.empty()) {
			// convert to 4d vector as required by model, and set as input
			blobFromImage(next, blob, 1.0, Size(672, 384));
			net.setInput(blob);
			Mat prob = net.forward();

			// get faces
			vector<Rect> faces;
			std::vector<float> confidences;
			int looking = 0;
			float* data = (float*)prob.data;
			for (size_t i = 0; i < prob.total(); i += 7)
			{
				float confidence = data[i + 2];
				if (confidence > confidenceFace)
				{
					int left = (int)(data[i + 3] * frame.cols);
					int top = (int)(data[i + 4] * frame.rows);
					int right = (int)(data[i + 5] * frame.cols);
					int bottom = (int)(data[i + 6] * frame.rows);
					int width = right - left + 1;
					int height = bottom - top + 1;

					faces.push_back(Rect(left, top, width, height));
					confidences.push_back(confidence);
				}
			}
			//int detSentiment;
			map<Sentiment, int> sent = {
				{Neutral, 0},
				{Happy, 0},
				{Confused, 0},
				{Surprised, 0},
				{Anger, 0},
				{Unknown, 0}
			};
			// look for poses
			for(auto const& r: faces) {
				// make sure the face rect is completely inside the main Mat
				if ((r & Rect(0, 0, next.cols, next.rows)) != r) {
					continue;
				}

				std::vector<Mat> outs;
				std::vector<String> names{"angle_y_fc", "angle_p_fc", "angle_r_fc"};
				cv::Mat face = next(r);

				// convert to 4d vector, and process thru neural network
				blobFromImage(face, poseBlob, 1.0, Size(60, 60));
				posenet.setInput(poseBlob);
				posenet.forward(outs, names);
				poseChecked = true;
				// the shopper is looking if their head is tilted within a 45 degree angle relative to the shelf
				if ( (outs[0].at<float>(0) > -22.5) && (outs[0].at<float>(0) < 22.5) &&
						(outs[1].at<float>(0) > -22.5) && (outs[1].at<float>(0) < 28.5) ) {
					looking++;
				}

				// convert to 4d vector, and propagate through sentiment Neural Network
				blobFromImage(face, sentBlob, 1.0, Size(64, 64));
				sentnet.setInput(sentBlob);
				Mat prob = sentnet.forward();
				sentChecked = true;

				// flatten the result from [1, 5, 1, 1] to [1, 5]
				Mat flat = prob.reshape(1, 5);
				// Find the max in returned list of sentiments
				Point maxLoc;
				double confidence;
				minMaxLoc(flat, 0, &confidence, 0, &maxLoc);
				Sentiment s;
				if (confidence > static_cast<double>(confidenceMood)) {
					s = static_cast<Sentiment>(maxLoc.y);
				} else {
					s = Unknown;
				}
				sent[s] = sent.at(s) + 1;
			}
			ClassroomInfo info;
			info.students = faces.size();
			info.sent = sent;
			info.lookers = looking;
			updateInfo(info);
		}
	}
}

// Reset curret sent pose data 
void resetCurrentInfo()
{
	m2.lock();
	ClassroomInfo rtn = currentInfo;
	currentInfo.students = 0;
	currentInfo.lookers = 0;
	for (pair<Sentiment, int> element : currentInfo.sent) {
		Sentiment s = element.first;		
		currentInfo.sent[s] = 0;
	}
	m2.unlock();
}

// 
void resetData() {
	while (keepRunning.load()) {
		resetCurrentInfo();
		this_thread::sleep_for(chrono::seconds(2));
	}
}

// signal handler for the main thread
void handle_sigterm(int signum)
{
	/* we only handle SIGTERM and SIGKILL here */
	if (signum == SIGTERM) {
		cout << "Interrupt signal (" << signum << ") received" << endl;
		sig_caught = 1;
	}
}

// Replace .xml with .bin
void replaceWithExt(string& s, const string& newExt) {
	string::size_type i = s.rfind('.', s.length());
	if (i != string::npos) {
		s.replace(i+1, newExt.length(), newExt);
	}
}
// Getting class name 
void getclassName()   
{
	string timeTable="/opt/intel/openvino/inference_engine/samples/classroom_analytics/timetable.txt";
	std::time_t t_old = std::time(0);   // get time now
	std::tm* now_old = std::localtime(&t_old);
	int Present_Time = now_old->tm_hour;
	char *Token ;
	string className;
	char classTime[20]="";
	char delim[10]=" ";
	int i=1;
        //std::ifstream is;
	std::string classtimeAndName;
	std::ifstream inputfile(timeTable);

	if (inputfile.is_open()){
		inputfile.seekg (0, inputfile.end);
		int length = inputfile.tellg();
		inputfile.seekg (0, inputfile.beg); 
		if(length == 0) {
			std::cout << "timetable file is empty" << std::endl;
			inputfile.close();
			exit(0); 
		}

		while(getline(inputfile, classtimeAndName)){
			char* buf = strdup(classtimeAndName.c_str());
			classtimeAndName.clear();
			Token = strtok (buf,delim);
			while (Token != NULL) {
				if(++i==2) {
					className=Token;
				}     
				else {  
					strcpy(classTime,Token);
				}
				Token = strtok (NULL, " ");
			}
			if(stoi(classTime) == Present_Time) {
				memset(checkTime,0,sizeof(checkTime));
				subject=className;
				sprintf(checkTime,"%s",classTime);
			}
			free(buf);
			i=1;
		}
	} else {
		std::cout << "timetable file does not exist" << std::endl;
		inputfile.close();
		exit(0);
	}

	inputfile.close();
}

using namespace InferenceEngine;

namespace {

	class Visualizer {
		private:
			cv::Mat frame_;
			const bool enabled_;
			cv::VideoWriter& writer_;
			float rect_scale_x_;
			float rect_scale_y_;
			static int const max_input_width_ = 1920;
			std::string const window_name_ = "Classroom Analytics demo";

		public:
			Visualizer(bool enabled, cv::VideoWriter& writer) : enabled_(enabled), writer_(writer) {}

			static cv::Size GetOutputSize(const cv::Size& input_size) {
				if (input_size.width > max_input_width_) {
					float ratio = static_cast<float>(input_size.height) / input_size.width;
					return cv::Size(max_input_width_, cvRound(ratio*max_input_width_));
				}
				return input_size;
			}

			void SetFrame(const cv::Mat& frame) {
				if (enabled_ || writer_.isOpened()) {
					frame_ = frame.clone();
					rect_scale_x_ = 1;
					rect_scale_y_ = 1;
					cv::Size new_size = GetOutputSize(frame_.size());
					if (new_size != frame_.size()) {
						rect_scale_x_ = static_cast<float>(new_size.height) / frame_.size().height;
						rect_scale_y_ = static_cast<float>(new_size.width) / frame_.size().width;
						cv::resize(frame_, frame_, new_size);
					}
				}
			}

			void Show() const {
				if (enabled_) {
					cv::imshow(window_name_, frame_);
				}
				if (writer_.isOpened()) {
					writer_ << frame_;
				}
			}

			void DrawObject(cv::Rect rect, const std::string& label_to_draw,
					const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
				if (enabled_ || writer_.isOpened()) {
					if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
						rect.x = cvRound(rect.x * rect_scale_x_);
						rect.y = cvRound(rect.y * rect_scale_y_);

						rect.height = cvRound(rect.height * rect_scale_y_);
						rect.width = cvRound(rect.width * rect_scale_x_);
					}
					cv::rectangle(frame_, rect, bbox_color);

					if (plot_bg && !label_to_draw.empty()) {
						int baseLine = 0;
						const cv::Size label_size =
							cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
						cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
								cv::Point(rect.x + label_size.width, rect.y + baseLine),
								bbox_color, cv::FILLED);
					}
					if (!label_to_draw.empty()) {
						putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
								text_color, 1, cv::LINE_AA);
					}
				}
			}

			void Finalize() const {
				cv::destroyWindow(window_name_);
				if (writer_.isOpened())
					writer_.release();
			}
	};

	const std::vector<std::string> actions_map = {"sitting", "standing", "raising_hand"};
	const int default_action_index = 0;  //  sitting

	std::string GetActionTextLabel(const unsigned label) {
		if (label < actions_map.size()) {
			return actions_map[label];
		}
		return "__undefined__";
	}

	float CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2) {
		int area1 = rect1.area();
		int area2 = rect2.area();

		float area_min = std::min(area1, area2);
		float area_intersect = (rect1 & rect2).area();

		return area_intersect / area_min;
	}

	cv::Rect DecreaseRectByRelBorders(const cv::Rect& r) {
		float w = r.width;
		float h = r.height;

		float left = std::ceil(w * 0);
		float top = std::ceil(h * 0);
		float right = std::ceil(w * 0);
		float bottom = std::ceil(h * .7);

		cv::Rect res;
		res.x = r.x + left;
		res.y = r.y + top;
		res.width = r.width - left - right;
		res.height = r.height - top - bottom;
		return res;
	}

	int GetIndexOfTheNearestPerson(const TrackedObject& face, const std::vector<TrackedObject>& tracked_persons) {
		int argmax = -1;
		float max_iom = std::numeric_limits<float>::lowest();
		for (size_t i = 0; i < tracked_persons.size(); i++) {
			float iom = CalculateIoM(face.rect, DecreaseRectByRelBorders(tracked_persons[i].rect));
			if ((iom > 0) && (iom > max_iom)) {
				max_iom = iom;
				argmax = i;
			}
		}
		return argmax;
	}

	std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
		std::map<int, int> face_track_id_to_label;
		for (const auto& track : face_tracks) {
			const auto& first_obj = track.first_object;
			// check consistency
			// to receive this consistency for labels
			// use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
			for (const auto& obj : track.objects) {
				SCR_CHECK_EQ(obj.label, first_obj.label);
				SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
			}

			auto cur_obj_id = first_obj.object_id;
			auto cur_label = first_obj.label;
			SCR_CHECK(face_track_id_to_label.count(cur_obj_id) == 0) << " Repeating face tracks";
			face_track_id_to_label[cur_obj_id] = cur_label;
		}
		return face_track_id_to_label;
	}

}  

int main(int argc, char* argv[]) 
{

	try {
		String model,Participation;
		String config,ad_weights_path,fr_weights_path,lm_weights_path,fd_weights_path;
		String sentmodel, posemodel,ad_model_path,fr_model_path,lm_model_path,fd_model_path;
		String sentconfig, poseconfig,fg_model_path;
    		String d_act,d_fd,d_lm,d_reid;
		string dbResp, connectionResp, classSection,influxdbIp;
		int noShow=0;

		std::vector<std::string> captured; // Identifying student names
		CommandLineParser parser(argc, argv, keys); 
		if(argc == 1 || parser.has("help")) {
			parser.printMessage();
			return -1;
		}

		auto video_path = parser.get<String>("input");
		config = parser.get<String>("config");
		backendId = parser.get<int>("backend");
		targetId = parser.get<int>("target");
		confidenceFace = parser.get<float>("faceconf");
		confidenceMood = parser.get<float>("moodconf");
		sentconfig = parser.get<String>("sentconfig");
		poseconfig = parser.get<String>("poseconfig");
		ad_model_path = parser.get<String>("persondetectionconfig");
		lm_model_path = parser.get<String>("landmarksregressionconfig");
		fr_model_path = parser.get<String>("facereidentificationconfig");
		fg_model_path = parser.get<String>("facegallerypath");
		fd_model_path = parser.get<String>("config");
		classSection = parser.get<String>("section");
		noShow       = parser.get<int>("noshow");
		d_act        = parser.get<String>("d_act");
		d_fd         = parser.get<String>("d_fd");
		d_lm         = parser.get<String>("d_lm");
		d_reid       = parser.get<String>("d_reid");
		influxdbIp   = parser.get<String>("influxip");

		model = config;
		replaceWithExt(model, "bin");
		sentmodel = sentconfig;
		replaceWithExt(sentmodel, "bin");
		posemodel = poseconfig;
		replaceWithExt(posemodel, "bin");
		ad_weights_path = ad_model_path;
		replaceWithExt(ad_weights_path, "bin");
		lm_weights_path = lm_model_path;
		replaceWithExt(lm_weights_path, "bin");
		fr_weights_path = fr_model_path;
		replaceWithExt(fr_weights_path, "bin");
		fd_weights_path = fd_model_path;
		replaceWithExt(fd_weights_path, "bin");

		std::map<std::string, InferencePlugin> plugins_for_devices;
		std::vector<std::string> devices = {d_act, d_fd, d_lm,
			d_reid};

		std::string dbCreate = "curl -i -XPOST http://" + influxdbIp +":8086/query --data-urlencode "+"\""+"q=CREATE DATABASE Analytics"+"\"" +" >/dev/null 2>&1";
		int ret = system(dbCreate.c_str());
		if(ret != 0) {
			std::cout <<"Failed to connect to DB : ret val "<< ret << std::endl;
			exit(0);
		}

		//getting student name from face gallary.json
		std::vector<std::string> students;
		students = getStudentNames(fg_model_path);

		for (const auto &device : devices) {
			if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
				continue;
			}
			slog::info << "Loading plugin " << device << slog::endl;
			InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device);
			printPluginVersion(plugin, std::cout);
			/** Load extensions for the CPU plugin **/
			if ((device.find("CPU") != std::string::npos)) {
				plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
				if (!FLAGS_l.empty()) {
					// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
					auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
					plugin.AddExtension(extension_ptr);
					slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
				}
			} else if (!FLAGS_c.empty()) {
				// Load Extensions for other plugins not CPU
				plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
			}
			plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
			if (FLAGS_pc)
				plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
			plugins_for_devices[device] = plugin;
		}

		// Load action detector
		ActionDetectorConfig action_config(ad_model_path, ad_weights_path);
		action_config.plugin = plugins_for_devices[d_act];
		action_config.is_async = true;
		action_config.enabled = !ad_model_path.empty();
		action_config.detection_confidence_threshold = FLAGS_t_act;
		ActionDetection action_detector(action_config);

		// Load face detector
		detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
		face_config.plugin = plugins_for_devices[d_fd];
		face_config.is_async = true;
		face_config.enabled = !fd_model_path.empty();
		face_config.confidence_threshold = FLAGS_t_fd;
		face_config.input_h = FLAGS_inh_fd;
		face_config.input_w = FLAGS_inw_fd;
		face_config.increase_scale_x = FLAGS_exp_r_fd;
		face_config.increase_scale_y = FLAGS_exp_r_fd;
		detection::FaceDetection face_detector(face_config);

		// Load face reid
		CnnConfig reid_config(fr_model_path, fr_weights_path);
		reid_config.max_batch_size = 16;
		reid_config.enabled = face_config.enabled && !fr_model_path.empty() && !lm_model_path.empty();
		reid_config.plugin = plugins_for_devices[d_reid];
		VectorCNN face_reid(reid_config);

		// Load landmarks detector
		CnnConfig landmarks_config(lm_model_path, lm_weights_path);
		landmarks_config.max_batch_size = 16;
		landmarks_config.enabled = face_config.enabled && reid_config.enabled && !lm_model_path.empty();
		landmarks_config.plugin = plugins_for_devices[d_lm];
		VectorCNN landmarks_detector(landmarks_config);

		// Create face gallery
		EmbeddingsGallery face_gallery(fg_model_path, FLAGS_t_reid, landmarks_detector, face_reid);
		//EmbeddingsGallery face_gallery(FLAGS_fg, FLAGS_t_reid, landmarks_detector, face_reid);

		// Create tracker for reid
		TrackerParams tracker_reid_params;
		tracker_reid_params.min_track_duration = 1;
		tracker_reid_params.forget_delay = 150;
		tracker_reid_params.affinity_thr = 0.8;
		tracker_reid_params.averaging_window_size_for_rects = 1;
		tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
		tracker_reid_params.drop_forgotten_tracks = false;
		tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
		tracker_reid_params.objects_type = "face";

		Tracker tracker_reid(tracker_reid_params);

		// Create Tracker for action recognition
		TrackerParams tracker_action_params;
		tracker_action_params.min_track_duration = 8;
		tracker_action_params.forget_delay = 150;
		tracker_action_params.affinity_thr = 0.95;
		tracker_action_params.averaging_window_size_for_rects = 5;
		tracker_action_params.bbox_heights_range = cv::Vec2f(10, 1080);
		tracker_action_params.drop_forgotten_tracks = false;
		tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
		tracker_action_params.objects_type = "action";

		Tracker tracker_action(tracker_action_params);

		//cv::Mat frame, prev_frame;
		cv::Mat prev_frame;
		DetectedActions actions;
		detection::DetectedObjects faces;

		float total_time_ms = 0.f;
		size_t num_frames = 0;
		const char ESC_KEY = 27;
		const cv::Scalar red_color(0, 0, 255);
		const cv::Scalar green_color(0, 128, 0);
		const cv::Scalar white_color(255, 255, 255);
		std::vector<std::map<int, int>> face_obj_id_to_action_maps;

		slog::info << "Reading video '" << video_path << "'" << slog::endl;
		ImageGrabber cap(video_path);
		if (!cap.IsOpened()) {
			slog::err << "Cannot open the video" << slog::endl;
			return 1;
		}

		if (cap.GrabNext()) {
			cap.Retrieve(frame);
		} else {
			slog::err << "Can't read the first frame" << slog::endl;
			return 1;
		}

		action_detector.enqueue(frame);
		action_detector.submitRequest();
		face_detector.enqueue(frame);
		face_detector.submitRequest();
		prev_frame = frame.clone();

		bool is_last_frame = false;
		auto prev_frame_path = cap.GetVideoPath();

		cv::VideoWriter vid_writer;
		if (!FLAGS_out_v.empty()) {
			vid_writer = cv::VideoWriter(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
					cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
		}
		Visualizer sc_visualizer(!FLAGS_no_show, vid_writer);

		if (!FLAGS_no_show) {
			std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
		}
		// open face model
		net = readNet(model, config);
		net.setPreferableBackend(backendId);
		net.setPreferableTarget(targetId);

		// open pose model
		sentnet = readNet(sentmodel, sentconfig);
		sentnet.setPreferableBackend(backendId);
		sentnet.setPreferableTarget(targetId);

		// open video capture source
		// open pose model
		posenet = readNet(posemodel, poseconfig);
		posenet.setPreferableBackend(backendId);
		posenet.setPreferableTarget(targetId);

		// initialize classroom info
		currentInfo.students = 0;
		currentInfo.lookers = 0;
		currentInfo.sent = {
			{Neutral, 0},
			{Happy, 0},
			{Confused, 0},
			{Surprised, 0},
			{Anger, 0},
			{Unknown, 0}
		};

		signal(SIGTERM, handle_sigterm);
		// thread starts
		thread t1(frameRunner);
		thread t2(resetData);

		while (!is_last_frame) {
			auto started = std::chrono::high_resolution_clock::now();
			is_last_frame = !cap.GrabNext();
			if (!is_last_frame)
				cap.Retrieve(frame);
			getclassName();    
			addImage(frame);

			std::vector<std::string> students;
			students = getStudentNames(fg_model_path);

			sc_visualizer.SetFrame(prev_frame);
			face_detector.wait();
			face_detector.fetchResults();
			faces = face_detector.results;

			action_detector.wait();
			action_detector.fetchResults();
			actions = action_detector.results;

			if (!is_last_frame) {
				prev_frame_path = cap.GetVideoPath();
				//cap.GetFrameIndex();
				face_detector.enqueue(frame);
				face_detector.submitRequest();
				action_detector.enqueue(frame);
				action_detector.submitRequest();
			}
			std::vector<cv::Mat> face_rois, landmarks, embeddings;
			TrackedObjects tracked_face_objects;

			for (const auto& face : faces) {
				face_rois.push_back(prev_frame(face.rect));
			}
			landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
			AlignFaces(&face_rois, &landmarks);
			face_reid.Compute(face_rois, &embeddings);
			auto ids = face_gallery.GetIDsByEmbeddings(embeddings);

			for (size_t i = 0; i < faces.size(); i++) {
				int label = ids.empty() ? EmbeddingsGallery::unknown_id : ids[i];
				tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, label);
			}
			tracker_reid.Process(prev_frame, tracked_face_objects, num_frames);

			const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();

			TrackedObjects tracked_action_objects;
			for (const auto& action : actions) {
				tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
			}

			tracker_action.Process(prev_frame, tracked_action_objects, num_frames);
			const auto tracked_actions = tracker_action.TrackedDetectionsWithLabels();

			auto elapsed = std::chrono::high_resolution_clock::now() - started;
			auto elapsed_ms =
				std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

			total_time_ms += elapsed_ms;
			num_frames += 1;

			std::map<int, int> frame_face_obj_id_to_action;
			int participationCount=0; // standing count variable
			captured.clear();
			for (size_t j = 0; j < tracked_faces.size(); j++) 
			{
				const auto& face = tracked_faces[j];
				std::string label_to_draw;
				if (face.label != EmbeddingsGallery::unknown_id)
				{
					label_to_draw += face_gallery.GetLabelByID(face.label);
					captured.push_back(label_to_draw);
				}
				label_to_draw =label_to_draw.substr(label_to_draw.find("_")+1);
				sc_visualizer.DrawObject(face.rect, label_to_draw, green_color , white_color, true);

				int person_ind = GetIndexOfTheNearestPerson(face, tracked_actions);
				int action_ind = default_action_index;
				if (person_ind >= 0) {
					action_ind = tracked_actions[person_ind].label;
				}

				label_to_draw += "(" + GetActionTextLabel(action_ind) + ")";
				frame_face_obj_id_to_action[face.object_id] = action_ind;

				Participation = GetActionTextLabel(action_ind);
				if((Participation == (char *)"standing")||(Participation == (char *)"raising_hand"))
					participationCount++;
				Participation.clear();
				sc_visualizer.DrawObject(face.rect, label_to_draw, green_color, white_color, true);
				
			}
			
                        face_obj_id_to_action_maps.push_back(frame_face_obj_id_to_action);

			string label;
			ClassroomInfo info = getCurrentInfo();

			double attentiveStudents = 0, totalStudents = 0;

			attentiveStudents = info.lookers;
			totalStudents = info.students;

			label = format("Students: %d,Neutral: %d,Happy: %d,Confused: %d,Surprised: %d,Anger: %d,Unknown: %d",
					info.students, info.sent[Neutral], info.sent[Happy], info.sent[Confused],
					info.sent[Surprised], info.sent[Anger], info.sent[Unknown]);
			putText(frame, label, Point(0, 20), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255, 255));
			int nonLookers = info.students - info.lookers;
			label = format("Attentive: %d, Non-Attentive: %d",
					info.lookers, nonLookers);
			putText(frame,label,Point(0,50),FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255,255));
			totalEmotions= info.sent[Neutral] + info.sent[Happy] + info.sent[Confused] + info.sent[Surprised] + info.sent[Anger]
				+ info.sent[Unknown];
			happinessEmotions =  info.sent[Happy];
			double happinessIndex = 0, attentiveIndex = 0,participationIndex = 0;

			Mat nex = nextImageAvailable();
			if (nex.empty()) { 
				if (info.students > 0) {
					happinessIndex = happinessCal(totalEmotions,happinessEmotions);
					attentiveIndex = attentiveCal(totalStudents,attentiveStudents);
					participationIndex = participationCal(static_cast<double>(participationCount),static_cast<double>(info.students));
					label = format("Attentivity Index: %.2f", attentiveIndex);
					putText(frame,label,Point(0,80),FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255),1);
					label = format("Happiness Index: %.2f", happinessIndex);
					putText(frame,label,Point(0,110),FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255,255));
					label = format("Participation Index: %.2f", participationIndex);
					putText(frame,label,Point(0,140),FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255,255));
				}
			}
			if (waitKey(delay) == 27 || sig_caught) {
				cout << "Attempting to stop background threads" << endl;
				keepRunning = false;
				break;
			}

			if(subject.length()==0)	
				subject = "No_Time_Assigned";

			//Inserting data in to infuxdb
			std::string commonPath = "curl -i -XPOST 'http://" + influxdbIp +":8086/write?db=Analytics' --data-binary '" + classSection;

			std::string classData = commonPath +',' + "classname="+ subject + " " + "studentpresent="+ std::to_string(info.students) + ',' +"participation=" + std::to_string(participationIndex) + ','+ "happiness=" + std::to_string(happinessIndex) +','+"attentivity=" + std::to_string(attentiveIndex) +"'" +" >/dev/null 2>&1";
			int  writeStatus = system(classData.c_str());
			if (writeStatus != 0) {
				std::cout <<"Failed to insert data in DB : ret val" << writeStatus << std::endl;
				classData.clear();
				exit(0);
			}  
			classData.clear();
			commonPath.clear();
			// For Finding Faces 
			std::string str;
			int i=0,j=0;

			std::time_t t_old = std::time(0);   // get time now
			std::tm* now_old = std::localtime(&t_old);
			int hour = now_old->tm_hour;
			int sec = now_old->tm_sec;
			int min=now_old->tm_min;
			if(atoi(checkTime) == hour && min==30 && (sec == 1 || sec == 2)) {
				// Inserting absent List into Database for every one hour
				std::string studentName;
				for (const auto &name : students) {
					studentName = name;
					i=0,j=0;
					while(i < static_cast<int>(captured.size())) {
						if(studentName == captured[i]) {
							j++;
							break;
						}
						i++;
					}
					if(j==0) {
						std::string name = studentName.substr(3);
						std::string commonPath = "curl -i -XPOST 'http://" + influxdbIp +":8086/write?db=Analytics' --data-binary 'AbsentList,";
						std::string absentBuffer = commonPath + "subjectName=" + subject+','+  "section="+ classSection +" " + "absentName=" +"\"" + name + "\"" + "'" +" >/dev/null 2>&1";
						int  writeStatus = system(absentBuffer.c_str());
						if (writeStatus != 0) {
							std::cout <<"Failed to insert absenties data in DB : ret val" << writeStatus << std::endl;
							absentBuffer.clear();
							exit(0);
						} 
						absentBuffer.clear();
						commonPath.clear();
					}
				}	
			}
			students.clear(); 
			attentiveStudents = totalStudents = totalEmotions = happinessEmotions = 0;
			if(noShow!=1)
				sc_visualizer.Show();
			char key = cv::waitKey(1);
			if (key == ESC_KEY) {
				break;
			}
			if (FLAGS_last_frame >= 0 && num_frames > static_cast<size_t>(FLAGS_last_frame)) {
				break;
			}
			prev_frame = frame.clone();
		}
		t1.join(); 
		t2.join();
		sc_visualizer.Finalize();
		slog::info << slog::endl;
		float mean_time_ms = total_time_ms / static_cast<float>(num_frames);
		slog::info << "Mean FPS: " << 1e3f / mean_time_ms << slog::endl;
		slog::info << "Frames processed: " << num_frames << slog::endl;
		if (FLAGS_pc) {
			face_detector.wait();
			action_detector.wait();
			action_detector.PrintPerformanceCounts();
			face_detector.PrintPerformanceCounts();
			face_reid.PrintPerformanceCounts();
			landmarks_detector.PrintPerformanceCounts();
		}
		auto face_tracks = tracker_reid.vector_tracks();
		// correct labels for track
		std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_tracks);
		std::map<int, int> face_track_id_to_label = GetMapFaceTrackIdToLabel(new_face_tracks);

		DetectionsLogger logger(std::cout, FLAGS_r, FLAGS_ad);
		logger.DumpDetections(cap.GetVideoPath(), frame.size(), num_frames,
				new_face_tracks,
				face_track_id_to_label,
				actions_map, face_gallery.GetIDToLabelMap(),
				face_obj_id_to_action_maps);  
	}
	catch (const std::exception& error) {
		slog::err << error.what() << slog::endl;
		return 1;
	}
	catch (...) {
		slog::err << "Unknown/internal exception happened." << slog::endl;
		return 1;
	}
	slog::info << "Execution successful" << slog::endl;
	return 0;
}

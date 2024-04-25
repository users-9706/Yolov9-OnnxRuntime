#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;
using namespace Ort;
vector<string> readClassNames(const string& filename) {
    vector<string> classNames;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return classNames;
    }
    string line;
    while (getline(file, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    file.close();
    return classNames;
}
int main(int argc, char** argv)
{
    string filename = "coco.names";
    vector<string> labels = readClassNames(filename);
    Mat image = imread("bus.jpg");
    int ih = image.rows;
    int iw = image.cols;
    string onnxpath = "yolov9c.onnx";
    wstring modelPath = wstring(onnxpath.begin(), onnxpath.end());
    SessionOptions session_options;
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov9-c");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Session session_(env, modelPath.c_str(), session_options);
    vector<string> input_node_names;
    vector<string> output_node_names;
    size_t numInputNodes = session_.GetInputCount();
    size_t numOutputNodes = session_.GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << endl;
    }
    int output_h = 0;
    int output_w = 0;
    TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; 
    output_w = output_dims[2]; 
    cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << endl;
    int64 start = getTickCount();
    int w = image.cols;
    int h = image.rows;
    int _max = max(h, w);
    Mat image_ = Mat::zeros(Size(_max, _max), CV_8UC3);
    Rect roi(0, 0, w, h);
    image.copyTo(image_(roi));
    float x_factor = image_.cols / static_cast<float>(input_w);
    float y_factor = image_.rows / static_cast<float>(input_h);
    Mat blob = dnn::blobFromImage(image_, 1 / 255.0, Size(input_w, input_h), Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const array<const char*, 1> outNames = { output_node_names[0].c_str() };
    vector<Value> ort_outputs;
    try {
        ort_outputs = session_.Run(RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (exception e) {
        cout << e.what() << endl;
    }
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    Mat det_output = dout.t();
    session_options.release();
    session_.release();
    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++) {
        Mat classes_scores = det_output.row(i).colRange(4, 84);
        Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        if (score > 0.25)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }
    vector<int> indexes;
    dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        rectangle(image, boxes[index], Scalar(0, 0, 255), 2, 8);
        rectangle(image, Point(boxes[index].tl().x, boxes[index].tl().y - 20),
            Point(boxes[index].br().x, boxes[index].tl().y), Scalar(0, 255, 255), -1);
        putText(image, labels[idx], Point(boxes[index].tl().x, boxes[index].tl().y), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, 8);
    }
    float t = (getTickCount() - start) / static_cast<float>(getTickFrequency());
    putText(image, format("FPS: %.2f", 1 / t), Point(20, 40), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, 8);
    imshow("YOLOV9-ONNXRUNTIME", image);
    waitKey(0);
    return 0;
}

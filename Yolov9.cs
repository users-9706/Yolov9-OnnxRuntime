using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
namespace Yolov9
{
    class Program
    {
        private float sigmoid(float a)
        {
            float b = 1.0f / (1.0f + (float)Math.Exp(-a));
            return b;
        }
        public static string[] read_class_names(string path)
        {
            string[] class_names;
            List<string> str = new List<string>();

            StreamReader sr = new StreamReader(path);
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                str.Add(line);
            }
            class_names = str.ToArray();
            return class_names;
        }
        static void Main(string[] args)
        {
            string model_path = "yolov9-c.onnx";
            string image_path = "bus.jpg";
            float conf_threshold = 0.25f;
            float nms_threshold = 0.4f;
            Mat image = new Mat(image_path);
            Mat image_copy = image.Clone();
            string[] classes_names = read_class_names("coco.names");
            int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
            Mat max_image = Mat.Zeros(new OpenCvSharp.Size(max_image_length, max_image_length), MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.AppendExecutionProvider_CPU(0);
            InferenceSession onnx_session = new InferenceSession(model_path, options);
            Mat image_rgb = new Mat();
            Cv2.CvtColor(max_image, image_rgb, ColorConversionCodes.BGR2RGB);
            Mat resize_image = new Mat();
            Cv2.Resize(image_rgb, resize_image, new OpenCvSharp.Size(640, 640));
            long start = Cv2.GetTickCount();
            float[] result_array = new float[8400 * 84];
            Tensor<float> input_tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
            for (int y = 0; y < resize_image.Height; y++)
            {
                for (int x = 0; x < resize_image.Width; x++)
                {
                    input_tensor[0, 0, y, x] = resize_image.At<Vec3b>(y, x)[0] / 255f;
                    input_tensor[0, 1, y, x] = resize_image.At<Vec3b>(y, x)[1] / 255f;
                    input_tensor[0, 2, y, x] = resize_image.At<Vec3b>(y, x)[2] / 255f;
                }
            }
            List<NamedOnnxValue> input_ontainer = new List<NamedOnnxValue>();
            input_ontainer.Add(NamedOnnxValue.CreateFromTensor("images", input_tensor));
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result_infer = onnx_session.Run(input_ontainer);
            DisposableNamedOnnxValue[] results_onnxvalue = result_infer.ToArray();
            Tensor<float> result_tensors = results_onnxvalue[0].AsTensor<float>();
            result_array = result_tensors.ToArray();
            onnx_session.Dispose();
            resize_image.Dispose();
            image_rgb.Dispose();
            List<string> classes = new List<string>();
            List<float> scores = new List<float>();
            List<Rect> rects = new List<Rect>();
            Mat result_data = new Mat(84, 8400, MatType.CV_32F, result_array);
            result_data = result_data.T();
            float[] factors = new float[2];
            factors = new float[2];
            factors[0] = factors[1] = (float)(max_image_length / 640.0);
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = result_data.Row(i).ColRange(4, 84);
                Point max_classId_point, min_classId_point;
                double max_score, min_score;
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);
                if (max_score > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * factors[0]);
                    int y = (int)((cy - 0.5 * oh) * factors[1]);
                    int width = (int)(ow * factors[0]);
                    int height = (int)(oh * factors[1]);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;
                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                }
            }
            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, conf_threshold, nms_threshold, out indexes);
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                Rect box = position_boxes[index];
                Cv2.Rectangle(image, position_boxes[index], new Scalar(0, 0, 255), 2, LineTypes.Link8);
                Cv2.Rectangle(image, new Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y - 20),
                    new Point(position_boxes[index].BottomRight.X, position_boxes[index].TopLeft.Y), new Scalar(0, 255, 255), -1);
                Console.WriteLine(classes_names[class_ids[index]]);
                Cv2.PutText(image, classes_names[class_ids[index]],new Point(position_boxes[index].X, position_boxes[index].Y - 5),
                    HersheyFonts.HersheyPlain, 2, new Scalar(255, 0, 0), 2);
            }
            float t = ((float)(Cv2.GetTickCount() - start)) / ((float)Cv2.GetTickFrequency());
            Cv2.PutText(image, string.Concat("FPS:", (1/t).ToString("0.00")), new Point(20, 40), HersheyFonts.HersheyPlain, 2, new Scalar(255, 0, 0), 2);
            Cv2.ImShow("YOLOV9-ONNXRUNTIME", image);
            Cv2.WaitKey(0);
        }
    }
}

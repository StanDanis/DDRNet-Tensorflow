import cv2
import numpy as np
from imutils.video import FileVideoStream
from imutils.video import FPS
from threading import Thread
from timeit import default_timer as timer
import time
import psutil
import GPUtil
import onnxruntime as rt 
import onnx

class FileVideoStreamShow:
    """class start video showing process for opencv cv.imshow"""
    def __init__(self, frame=None, window_name='video', max_num_f=None):
        self.frame = frame
        self.stopped = False
        self.window_name = window_name
        
        self.thread = Thread(target=self.show, args=())
        self.thread.daemon = True

        self.count = 0
        self.max_num_f = max_num_f
        
    def start(self):
        self.thread.start()
        return self

    def show(self):
        while not self.stopped:
            if self.frame is not None:
                frame = self.frame
            else:
                frame = np.zeros((400, 400, 1), dtype='float32')
            cv2.imshow(self.window_name, frame)
                
            if cv2.waitKey(1) == ord("q") or (self.max_num_f == self.count):
                self.stopped = True

    def stop(self):
        self.stopped = True

class Measure:
    """class measure CPU, GPU and RAM usage"""
    def __init__(self) -> None:
        self.thread = Thread(target=self.measure_all, args=())
        self.thread.daemon = True

        self.stopped = False
        self.cpu, self.memory, self.gpu_stats_dic = [], [], {}

    def start(self):
        self.thread.start()
        return self

    def cpu_memory_stats(self):
        return psutil.cpu_percent(interval=0.2, percpu=True), psutil.virtual_memory()

    def gpu_stats(self):
        tmp = GPUtil.getGPUs()[0].temperature
        gpu_mem_tot = GPUtil.getGPUs()[0].memoryTotal
        gpu_mem_util = GPUtil.getGPUs()[0].memoryUtil
        gpu_load = GPUtil.getGPUs()[0].load

        return {'load': gpu_load, 'gpu_mem_util': gpu_mem_util, 'gpu_mem_tot': gpu_mem_tot,
                'tmp': tmp}

    def measure_all(self):
        while not self.stopped:
            self.cpu , self.memory = self.cpu_memory_stats()
            self.gpu_stats_dic = self.gpu_stats()

    def stop(self):
        self.stopped = True

class Evaluate:
    """class simulate idea app that use model, frequency of class (how fast show prediction) 
    is same as model inference time"""
    def __init__(self, video_path, model_definition, transform=None, queue_size=30, 
                meas=True, max_num_f=None):
        self.video_path = video_path
        self.model_definition = model_definition
        self.transform = transform
        self.queue_size = queue_size
        self.meas = meas

        self.fvs = FileVideoStream(self.video_path,  queue_size=self.queue_size, 
                                    transform=self.transform)
        self.fvs_show = FileVideoStreamShow(window_name=f'Evaluate- {self.video_path}', 
                                            max_num_f=max_num_f)    
        self.fps = FPS()

        if self.meas:
            self.measure_load = Measure()

        self.count = 0

        self.t1, self.t2 = [], []
        self.t1_model, self.t2_model = [], []
        self.cpu, self.memory, self.gpu = [], [], []

    def start_all_threads(self):
        print("[INFO] Starting video file read thread...")
        self.fvs.start()
        time.sleep(1.0)
        self.fps.start()
        time.sleep(1.0)
        print("[INFO] Starting video file show thread...")
        self.fvs_show.start()
        if self.meas:
            self.measure_load.start()
        return self

    def stop_all_threads(self):
        self.fvs.stop()
        time.sleep(1.0)
        self.fps.stop()
        time.sleep(1.0)
        self.fvs_show.stop()
        if self.meas:
            self.measure_load.stop()
        cv2.destroyAllWindows()
        return self

    def start_evaluate(self, main_loop_transform=None):
        print(f'[INFO] Press "q" to release video')
        self.start_all_threads()

        self.start = timer()

        while self.fvs.more():
            self.t1.append(timer())

            self.frame = self.fvs.read() 

            if self.frame is None or self.fvs_show.stopped:
                break

            self.t1_model.append(timer())        
            self.frame = self.model_definition(self.frame)
            # self.frame = np.uint8(((np.argmax(pred_img[0,:,:,:], axis=-1)/12)+0)*255)
            self.t2_model.append(timer())
          
            self.count = self.count + 1
            self.fvs_show.count = self.count
            if self.meas:
                self.cpu.append(self.measure_load.cpu)
                self.memory.append(self.measure_load.memory)
                self.gpu.append(self.measure_load.gpu_stats_dic)

            if main_loop_transform is not None:
                self.frame = main_loop_transform(self.frame)

            fps_v = np.round(self.count/(self.t1[-1] - self.start), 2)

            cv2.putText(self.frame, f"Queue Size: {self.fvs.Q.qsize()} frame n.: {self.count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(self.frame, f"FPS: {fps_v}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.fvs_show.frame = self.frame 
            self.t2.append(timer())
            
            self.fps.update()

        self.stop_all_threads()
        print(f"[INFO] Elasped time: {self.fps.elapsed():.2f}")
        print(f"[INFO] Number of frames: {self.count:.2f}")
        print(f"[INFO] Number of frames from fps: {self.fps.elapsed()*self.fps.fps():.2f}")
        print(f"[INFO] Approx. FPS: {self.fps.fps():.2f}")

class Onnx_model:
    def __init__(self, model=None, names=None):
        self.sess = model
        self.names = names
        
    def get_predict(self, frame):
        if frame is not None:
            return self.sess.run(self.names, {'x': frame})[0]

def transform_func(img=None):
    if img is not None:
        
        frame = cv2.resize(img, (400, 400), cv2.INTER_NEAREST)/255
        frame = np.expand_dims(frame, 0).astype('float32')
        
        return frame
    else:
        return None

if __name__ == '__main__':

    model_name = "onnx/model400.onnx"
    video_name = 'IMG_7678k.mp4'
    providers = ['CUDAExecutionProvider']

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = rt.InferenceSession(model_name, sess_options=sess_options,
                            providers=providers) 

    output_names = [n.name for n in onnx.load(model_name).graph.output]

    onnx_model = Onnx_model(model=sess, names=output_names).get_predict

    eval_his = Evaluate(video_name, onnx_model, transform_func, meas=True)
    eval_his.start_evaluate()










    
    



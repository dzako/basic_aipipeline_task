import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from datasets import load_dataset, Dataset, Image
import math
from aux import *

if __name__ == "__main__":
    # the main pipeline start
    # should be split into sub-modules , but didn't have time for refactoring

    args = parse_args()
    datasetn = args.dataset
    torch.set_num_threads(args.num_threads)

    # load yolo v5 model
    networkn = args.network
    model = torch.hub.load("ultralytics/yolov5", "custom", networkn)
    # checked this is faster, but hard to profile
    # model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.onnx")
    model.eval()
    model = model.to("cpu").float()

    # loading dataset
    t1 = time.perf_counter(), time.process_time()
    # check if local dataset
    if "local_dataset" in datasetn:
        di = {
            "image": [
                os.path.join(datasetn, f)
                for f in os.listdir(datasetn)
                if os.path.isfile(os.path.join(datasetn, f))
            ]
        }

        dataset = Dataset.from_dict(di).cast_column("image", Image())
    else:
        # otherwise pull dataset directly from huggingf and use only args.maxim images
        dataset = load_dataset(datasetn, split=f"train[:{args.maxim}]")

    print("loaded dataset:")
    print(dataset)
    print(dataset.num_rows)
    t2 = time.perf_counter(), time.process_time()

    # create dir for outputs
    outimg = f"{args.dataset.replace('/', '')}_{networkn}_out"
    outprofile = f"{args.dataset.replace('/', '')}_{networkn}_profile"
    if not os.path.exists(outimg):
        os.makedirs(outimg)
    if not os.path.exists(outprofile):
        os.makedirs(outprofile)

    with open(os.path.join(outprofile, "general_profile.txt"), "w") as f:
        f.write(str(dataset) + "\n" + datasetn + "\n")
        f.write(f" loading data real time: {t2[0] - t1[0]:.2f} \n")
        f.write(f" loading data CPU time: {t2[1] - t1[1]:.2f} seconds\n")

    # number of batches total
    batches = math.ceil(dataset.num_rows / args.batch)

    t1 = time.perf_counter(), time.process_time()
    bresults = []
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for b in range(batches):
                    print(f"processing batch {b}")
                    endi = (b + 1) * args.batch
                    if endi > dataset.num_rows:
                        endi = dataset.num_rows
                    batch_images = dataset[(b * args.batch) : endi]["image"]
                    results = model(batch_images, size=args.imsize, profile=True)
                    bresults.append(results)

    t2 = time.perf_counter(), time.process_time()
    print(f"network inference total time {t2[0] - t1[0]:.2f} seconds")

    # save profile data
    with open(os.path.join(outprofile, "network_profile.txt"), "w") as f:
        f.write(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=100
            )
        )

    with open(os.path.join(outprofile, "general_profile.txt"), "a") as f:
        f.write(f" inferring network real time: {t2[0] - t1[0]:.2f} seconds\n")
        f.write(f" inferring network CPU time: {t2[1] - t1[1]:.2f} seconds\n")

    # parse runtimes
    runtime_pre = 0
    runtime_network = 0
    runtime_post = 0

    open(os.path.join(outprofile, "detection_output.txt"), "w").close()
    for n, r in enumerate(bresults):
        with open(os.path.join(outprofile, "detection_output.txt"), "a") as f:
            f.write(f"images batch {n}\n")
            f.write(str(results) + "\n")
        # dump the detection data
        runtime_pre += r.t[0]
        runtime_network += r.t[1]
        runtime_post += r.t[2]

    # save yolo network total times
    with open(os.path.join(outprofile, "yolo_profile.txt"), "w") as f:
        f.write(str(dataset) + "\n" + datasetn + "\n")
        f.write(f"total real preprocess: {runtime_pre * args.batch / 1000.}secs\n")
        f.write(
            f"total real network inference: {runtime_network * args.batch / 1000.}secs\n"
        )
        f.write(f"total real postprocess: {runtime_post * args.batch / 1000.}secs\n")

    # the last stage dumping files with object detections
    # this can be made optimal
    t1 = time.perf_counter(), time.process_time()
    print(f"Dumping the labelled images to {outimg}")
    for i in range(dataset.num_rows):
        bi = math.floor(i / args.batch)
        wbi = i % args.batch
        # print(i, bi, wbi)
        curi = dataset[i]["image"]
        if "image_id" in dataset.features:
            # the dataset if from huggingf , using image_ids as names
            filename = str(dataset[i]["image_id"])
        else:
            filename = curi.filename.split("/")[-1]
        to_img(
            outimg,
            filename,
            curi,
            bresults[bi].xyxy[wbi],
        )
    t2 = time.perf_counter(), time.process_time()

    with open(os.path.join(outprofile, "general_profile.txt"), "a") as f:
        f.write(f" dumping labelled images real time: {t2[0] - t1[0]:.2f} seconds\n")
        f.write(f" dumping labelled images CPU time: {t2[1] - t1[1]:.2f} seconds\n")

from io import BytesIO
from typing import Optional
import zipfile
from pymilvus import MilvusClient
from fastapi import FastAPI, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
from dotenv import load_dotenv
import os

from type import Pose
from utils import (
    convert_bytes_to_np_array,
    convert_image_to_np_array,
)
from schema import face_schema
from http import HTTPStatus

load_dotenv()

# env
face_collection = os.getenv("face_embedding", "placeholder")

# milvus client
milvusClient = MilvusClient("./milvus.db")
if not milvusClient.has_collection(face_collection):
    milvusClient.create_collection(
        collection_name=face_collection, schema=face_schema, dimension=512
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the face analysis model
rec_model = FaceAnalysis(providers=["CPUExecutionProvider"])
rec_model.prepare(ctx_id=0, det_size=(640, 640))


@app.get("/reset/{userId}")
async def reset(userId: str):
    milvusClient.delete(collection_name=face_collection, filter=f'code == "{userId}"')
    return JSONResponse(
        status_code=HTTPStatus.OK, content={"message": "Delete success"}
    )


@app.get("/missing-pose/{userId}")
async def missing_pose(userId: str):
    # get existed faces
    existed_face_list = milvusClient.query(
        collection_name=face_collection,
        filter=f'code == "{userId}"',
        output_fields=["id"],
    )

    missing_pose = len(Pose) - len(existed_face_list)
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"missingPose": missing_pose},
    )


@app.post("/register")
async def register(
    userId: str = Form(...),
    img: UploadFile = Form(...),
):
    # unzip file
    # check if file is a zip file
    if not img.filename or not img.filename.endswith(".zip"):
        return JSONResponse(
            status_code=HTTPStatus.CONFLICT,
            content={"message": "Uploaded file must be a ZIP file."},
        )
    # unzip
    zip_bytes = await img.read()
    zip_file = zipfile.ZipFile(BytesIO(zip_bytes))

    # Extract all images from the zip
    image_files = [
        f for f in zip_file.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        return JSONResponse(
            status_code=HTTPStatus.CONFLICT,
            content={"message": "No image files found in ZIP."},
        )

    # check if the unzipped file contains 6 images
    existed_face_list = milvusClient.query(
        collection_name=face_collection,
        filter=f'code == "{userId}"',
        output_fields=["id"],
    )

    if len(existed_face_list) == len(Pose):
        return JSONResponse(
            status_code=HTTPStatus.CONFLICT, content={"message": "Đã có đủ hình ảnh!"}
        )

    # store it into db
    # Convert to vector
    for image_name in image_files:
        # Read each image inside ZIP
        image_data = zip_file.read(image_name)
        decodedImg = convert_bytes_to_np_array(image_data)

        # Convert to vector
        current_face = rec_model.get(decodedImg)[0]

        # Insert into DB
        new_record = {
            "code": userId,
            "vector": current_face.normed_embedding,
        }
        milvusClient.insert(collection_name=face_collection, data=new_record)

    # Return if success
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "message": "Thêm 6 tư thế thành công!",
        },
    )


@app.post("/verify")
async def verify_person(userId: str = Form(...), comparedImg: UploadFile = Form(...)):
    # get user face vectors
    face_vector_list = (
        milvusClient.query(
            collection_name=face_collection,
            filter=f'code == "{userId}"',
            output_fields=["vector"],
        ),
    )

    # unwrap the tuple
    if isinstance(face_vector_list, tuple):
        face_vector_list = face_vector_list[0]

    # convert to array
    decodedImg = convert_image_to_np_array(comparedImg)

    # Anti-spoofing and face validation using insightface
    try:
        # Detect faces with insightface
        detected_faces = rec_model.get(decodedImg)

        print(f"Number of faces detected: {len(detected_faces)}")

        # Check if no face detected
        if len(detected_faces) == 0:
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"message": "Không tìm thấy khuôn mặt!"},
            )

        # Check for multiple faces
        if len(detected_faces) > 1:
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={"message": "Không thể có nhiều hơn 1 khuôn mặt!"},
            )

        face = detected_faces[0]

        # Get face bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1

        img_h, img_w = decodedImg.shape[:2]

        # Anti-spoofing Check 1: Face should not be too close (occupying too much of the image)
        face_ratio = face_width / img_w

        # Anti-spoofing Check 3: Detection score (confidence) - insightface provides det_score
        det_score = face.det_score if hasattr(face, "det_score") else 1.0
        print(f"Detection confidence: {det_score}")

        if det_score < 0.8:  # Low confidence detection - possible spoof
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={
                    "message": "Phát hiện khuôn mặt giả mạo! Vui lòng sử dụng khuôn mặt thật để chấm công"
                },
            )

        print(
            f"Anti-spoofing checks passed - confidence: {det_score}, face_ratio: {face_ratio:.2f}"
        )
        if face_ratio > 0.7:  # Face covers >70% of image width
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={"message": "Khuôn mặt quá gần camera! Vui lòng đứng xa hơn."},
            )

        # Anti-spoofing Check 2: Face should not be too small (likely a spoof or low quality)
        if face_ratio < 0.15:  # Face is less than 15% of image width
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={
                    "message": "Khuôn mặt quá xa hoặc quá nhỏ! Vui lòng đến gần hơn."
                },
            )

    except Exception as e:
        print(f"Face validation error: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"message": f"Lỗi kiểm tra khuôn mặt: {str(e)}"},
        )

    # Proceed with face recognition using insightface
    compared_face = detected_faces[0]

    if compared_face is not None:
        compared_face_embedding = compared_face.normed_embedding

        # Check if user has any registered faces
        if not face_vector_list or len(face_vector_list) == 0:
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"message": "Người dùng chưa đăng ký khuôn mặt!"},
            )

        # Compute similarity (cosine distance) to all target embeddings
        similarities = [
            np.dot(compared_face_embedding, e["vector"]) for e in face_vector_list
        ]

        # Take the maximum similarity
        max_similarity = max(similarities)
        print("Max similarity:", max_similarity)

        if max_similarity > 0.6:  # threshold (tune this)
            print("Target person recognized!")
            return JSONResponse(
                status_code=HTTPStatus.OK, content={"message": "Khuôn mặt trùng khớp!"}
            )
        else:
            print("Unknown person")
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={"message": "Khuôn mặt không trùng khớp!"},
            )
    else:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"message": "Không tìm thấy khuôn mặt!"},
        )


# @app.post("/pose")
# async def checkPost(res: Response, img: UploadFile = Form(...)):
#     # verify if there are faces
#     decodedImg = convert_image_to_np_array(img)
#     face_list = rec_model.get(decodedImg)
#
#     # return if multiple faces detected
#     if len(face_list) > 1:
#         return JSONResponse(
#             status_code=400, content={"message": "Không thể có nhiều hơn 1 khuôn mặt!"}
#         )
#
#     # Get the pose
#     face = face_list[0]
#     landmarks = face.landmark_2d_106
#     # show landmarks
#     # scale = 0.3  # shrink to 30% size
#     # resized = cv2.resize(decodedImg, None, fx=scale, fy=scale)
#     #
#     # for i, (x, y) in enumerate(landmarks):
#     #     x, y = int(x * scale), int(y * scale)  # scale landmarks too
#     #     cv2.putText(resized, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#     #
#     # cv2.imshow("Landmarks", resized)
#     # cv2.waitKey(0)
#     yaw, pitch, roll = compute_pose(decodedImg, landmarks)
#     pose = classify_pose(yaw, pitch, roll)
#     print("Pose:", pose)
#
#     return JSONResponse(status_code=200, content={"message": f"{pose}"})

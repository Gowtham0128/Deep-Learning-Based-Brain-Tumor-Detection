"""
DATA.py

Cleaned and modularized version of the original script.

This file provides:
- kmeans_color_quantization(image)
- preprocess helpers (resize, grayscale, denoise)
- binarize_and_mask(image)
- simple RPN-like segmentation helper
- DCNN training helper (small example)

It is intentionally self-contained and avoids Flask / DB calls. Use this as a starting point
to wire into a Flask app or a unit test harness.
"""

import os
import datetime
import numpy as np
import cv2
from PIL import Image
try:
	import imagehash  # type: ignore
except Exception:
	imagehash = None

# Keras may be provided either by the standalone `keras` package or via `tensorflow.keras`.
try:
	from keras.models import Sequential  # type: ignore
	from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
	from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore
except Exception:
	# fallback to tensorflow.keras (also add type: ignore so Pylance stops warning if its
	# interpreter is not set to the virtual environment). To fully resolve editor warnings,
	# select the workspace interpreter at m:/RY/.venv/Scripts/python.exe in VS Code.
	from tensorflow.keras.models import Sequential  # type: ignore
	from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
	from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore


def ensure_dirs():
	os.makedirs('static/dataset', exist_ok=True)
	os.makedirs('static/trained', exist_ok=True)
	os.makedirs('static/trained/bb', exist_ok=True)
	os.makedirs('static/trained/sg', exist_ok=True)
	os.makedirs('static/test', exist_ok=True)


def kmeans_color_quantization(image: np.ndarray, clusters: int = 8, rounds: int = 1) -> np.ndarray:
	"""Reduce colors in the image using k-means (OpenCV).

	Args:
		image: HxWxC uint8 image
		clusters: number of color clusters
		rounds: kmeans attempts

	Returns:
		quantized image (same shape, uint8)
	"""
	h, w = image.shape[:2]
	samples = image.reshape((-1, 3)).astype(np.float32)

	compactness, labels, centers = cv2.kmeans(
		samples,
		clusters,
		None,
		(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
		rounds,
		cv2.KMEANS_RANDOM_CENTERS,
	)
	centers = np.uint8(centers)
	res = centers[labels.flatten()]
	return res.reshape((h, w, 3))


def preprocess_resize_save(fname: str, size=(256, 256)) -> str:
	"""Resize an image from static/data and save to static/dataset.

	Returns saved filename path.
	"""
	src = os.path.join('static', 'data', fname)
	dst_dir = os.path.join('static', 'dataset')
	os.makedirs(dst_dir, exist_ok=True)
	dst = os.path.join(dst_dir, fname)
	img = cv2.imread(src)
	if img is None:
		raise FileNotFoundError(src)
	rez = cv2.resize(img, size)
	cv2.imwrite(dst, rez)
	return dst


def grayscale_and_denoise(fname: str) -> str:
	"""Create a grayscale and denoised version of the image and return path to denoised file."""
	src = os.path.join('static', 'data', fname)
	img = cv2.imread(src)
	if img is None:
		raise FileNotFoundError(src)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gpath = os.path.join('static', 'trained', f'g_{fname}')
	cv2.imwrite(gpath, gray)

	# read grayscale as BGR for fastNlMeans (works on color, but we'll denoise the original color)
	dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
	fname2 = f'ns_{fname}'
	dpath = os.path.join('static', 'trained', fname2)
	cv2.imwrite(dpath, dst)
	return dpath


def binarize_and_mask(fname: str, k_clusters: int = 4):
	"""Perform color quantization + adaptive thresholding and return result and mask arrays.

	Saves a binary threshold image to static/trained/bb/bin_<fname> and returns (result, mask)
	"""
	src = os.path.join('static', 'data', fname)
	image = cv2.imread(src)
	if image is None:
		raise FileNotFoundError(src)
	original = image.copy()
	kmeans = kmeans_color_quantization(image, clusters=k_clusters)

	gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY_INV, 21, 2)

	mask = np.zeros(original.shape[:2], dtype=np.uint8)
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	if len(cnts) == 0:
		# nothing found
		cv2.imwrite(os.path.join('static', 'trained', 'bb', f'bin_{fname}'), thresh)
		return None, None

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	# take largest contour
	c = cnts[0]
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
	cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

	result = cv2.bitwise_and(original, original, mask=mask)
	result[mask == 0] = (0, 0, 0)
	cv2.imwrite(os.path.join('static', 'trained', 'bb', f'bin_{fname}'), thresh)
	cv2.imwrite(os.path.join('static', 'trained', f'result_{fname}'), result)
	return result, mask


def rpn_segment(path_main='static/data'):
	"""Simple segmentation over all files in path_main; saves segments to static/trained/sg/"""
	out_dir = os.path.join('static', 'trained', 'sg')
	os.makedirs(out_dir, exist_ok=True)
	for fname in os.listdir(path_main):
		src = os.path.join(path_main, fname)
		img = cv2.imread(src)
		if img is None:
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		kernel = np.ones((3, 3), np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
		sure_bg = cv2.dilate(opening, kernel, iterations=3)
		dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
		ret2, sure_fg = cv2.threshold(dist_transform, 1.5 * dist_transform.max(), 255, 0)
		sure_fg = np.uint8(sure_fg)
		segment = cv2.subtract(sure_bg, sure_fg)
		seg_path = os.path.join(out_dir, f'sg_{fname}')
		cv2.imwrite(seg_path, segment)


def simple_tumour_mask(image_path: str) -> np.ndarray:
	"""Heuristic (demo-only) tumour-like mask generator.

	This uses grayscale + Otsu thresholding + morphological ops and returns a
	binary mask (uint8) with 255 where a candidate region exists. This is
	NOT medically accurate and only for demo overlaying.
	"""
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# invert if background is dark
	if np.mean(th) < 127:
		th = 255 - th
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
	opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

	# Keep the largest connected component as the candidate 'tumour'
	cnts = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	mask = np.zeros_like(opened)
	if not cnts:
		return mask
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	cv2.drawContours(mask, [cnts[0]], -1, 255, -1)
	return mask


def overlay_mask_on_image(image_path: str, mask_path: str = None, mask_array: np.ndarray = None,
						  out_path: str = None, color=(0, 0, 255), alpha: float = 0.4) -> str:
	"""Overlay a binary mask onto an image and save an annotated image.

	Args:
		image_path: path to input image (usually under static/data/)
		mask_path: optional path to a binary mask image (255 = mask)
		mask_array: optional numpy array mask (uint8) to use instead of mask_path
		out_path: optional output path; if None, writes to static/trained/annot_<fname>
		color: BGR color tuple for overlay (default red)
		alpha: overlay alpha (0..1)

	Returns:
		path to the saved annotated image

	Notes:
		If neither mask_path nor mask_array is provided, this function will call
		`simple_tumour_mask` to produce a demo mask.
	"""
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(image_path)

	if mask_array is None:
		if mask_path is not None and os.path.exists(mask_path):
			m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			if m is None:
				raise FileNotFoundError(mask_path)
			mask = (m > 0).astype('uint8') * 255
		else:
			# fallback to heuristic mask generator
			mask = simple_tumour_mask(image_path)
	else:
		mask = (mask_array > 0).astype('uint8') * 255

	# make 3-channel overlay
	overlay = np.zeros_like(img, dtype=np.uint8)
	overlay[mask == 255] = color

	annotated = cv2.addWeighted(img, 1.0, overlay, alpha, 0)

	# draw mask contour as outline for clarity
	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	if cnts:
		cv2.drawContours(annotated, cnts, -1, (0, 255, 255), 2)

	# ensure output path
	base = os.path.basename(image_path)
	if out_path is None:
		out_dir = os.path.join('static', 'trained')
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f'annot_{base}')

	cv2.imwrite(out_path, annotated)
	return out_path


def build_small_cnn(input_shape=(128, 128, 3)):
	cnn = Sequential()
	cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	cnn.add(MaxPooling2D(pool_size=(2, 2)))
	cnn.add(Conv2D(32, (3, 3), activation='relu'))
	cnn.add(MaxPooling2D(pool_size=(2, 2)))
	cnn.add(Flatten())
	cnn.add(Dense(units=128, activation='relu'))
	cnn.add(Dense(units=1, activation='sigmoid'))
	cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return cnn


def dcnn_train(dataset_dir='dataset', epochs=2):
	"""Train a small CNN using ImageDataGenerator. This is an example helper.

	Expects dataset/training and dataset/test directories with class subfolders.
	"""
	train_dir = os.path.join(dataset_dir, 'training')
	test_dir = os.path.join(dataset_dir, 'test')
	if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
		raise FileNotFoundError('Please create dataset/training and dataset/test with images')

	train_gen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	train = train_gen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
	test = train_gen.flow_from_directory(test_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

	cnn = build_small_cnn()
	history = cnn.fit(train, steps_per_epoch=10, epochs=epochs, validation_data=test, validation_steps=5)
	# save model
	cnn.save(os.path.join('static', 'trained', 'model.h5'))
	return history


def predict_with_model(image_path, model=None):
	"""Load a saved model (if not provided) and predict a single image.

	Returns raw prediction scalar and binary decision (0/1).
	"""
	from keras.models import load_model

	if model is None:
		model_path = os.path.join('static', 'trained', 'model.h5')
		if not os.path.exists(model_path):
			raise FileNotFoundError('No trained model found. Run dcnn_train first.')
		model = load_model(model_path)

	img = load_img(image_path, target_size=(128, 128))
	arr = img_to_array(img) / 255.0
	arr = np.expand_dims(arr, axis=0)
	pred = model.predict(arr)
	return float(pred[0][0]), int(pred[0][0] >= 0.5)


def find_similar_by_hash(test_image_path, dataset_path='static/dataset', cutoff=1):
	"""Find first image in dataset with perceptual hash distance <= cutoff.

	Requires imagehash to be installed; returns matched filename or None.
	"""
	if imagehash is None:
		raise RuntimeError('imagehash is not available (install with pip install ImageHash Pillow)')
	h1 = imagehash.average_hash(Image.open(test_image_path))
	for fname in os.listdir(dataset_path):
		fpath = os.path.join(dataset_path, fname)
		try:
			h0 = imagehash.average_hash(Image.open(fpath))
			if (h0 - h1) <= cutoff:
				return fname
		except Exception:
			continue
	return None


def main_example():
	"""Small example run: ensure dirs, preprocess one file (if present) and run binarization."""
	ensure_dirs()
	data_dir = os.path.join('static', 'data')
	if not os.path.isdir(data_dir):
		print('Place images under static/data and re-run')
		return
	files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	if not files:
		print('No image files found under static/data')
		return
	fname = files[0]
	print('Processing', fname)
	preprocess_resize_save(fname)
	grayscale_and_denoise(fname)
	res, mask = binarize_and_mask(fname)
	if res is None:
		print('Binarization found no contours')
	else:
		print('Saved binarization and result images to static/trained')


if __name__ == '__main__':
	main_example()
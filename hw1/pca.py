import os
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from skimage import io

Q1_DIR = "../q1"
Q2_DIR = "../q2"


def parse_input():
	parser = argparse.ArgumentParser()

	parser.add_argument("--faces_dir", help="faces directory", default="../hw1_dataset")
	parser.add_argument("--face_npz", help="pre-read faces", default="hw1_face.npz")

	return parser.parse_args()


# imgs shape: N * feat
def pca(imgs, k=None):
	if k is None:
		k = imgs.shape[0]

	u, s, v = np.linalg.svd(imgs, full_matrices=False)
	return v[:k, :], s


def read_all_imgs(path="Aberdeen", save_path=None, test_index=list(range(7, 11))):
	img_shape = None
	train_imgs, test_imgs = [], []
	train_person, test_person = [], []
	for f in os.listdir(path):
		name, _ = os.path.splitext(f)
		person, index = name.split('_')

		f = os.path.join(path, f)
		img = io.imread(f)
		if img_shape is None:
			img_shape = img.shape

		img = img.flatten()
		if int(index) in test_index:
			test_imgs.append(img)
			test_person.append(int(person))
		else:
			train_imgs.append(img)
			train_person.append(int(person))

	test_imgs = np.asarray(test_imgs, dtype=np.float32)
	train_imgs = np.asarray(train_imgs, dtype=np.float32)
	test_person = np.asarray(test_person, dtype=np.uint8)
	train_person = np.asarray(train_person, dtype=np.uint8)

	if save_path is not None:
		np.savez(save_path, train_imgs, train_person, test_imgs, test_person, img_shape)
	return train_imgs, train_person, test_imgs, test_person, img_shape


def plot_face(face, path, shape):
	face = np.reshape(face, shape)
	face = face.astype(dtype=np.uint8)
	io.imsave(path, face)


def plot_eigens(eigenfaces, shape):	
	num_face, num_feats = eigenfaces.shape
	for fid in range(num_face):
		face = np.array(eigenfaces[fid, :])
		face -= np.min(face)
		face /= np.max(face)
		face *= 255

		path = os.path.join(Q1_DIR, "eigen_face%d.png" % fid)
		plot_face(face, path, shape)


def reconstruct_faces(face, eigenfaces, mean, num_eigen=4):
	face = face.astype(np.float32)
	reconst = np.dot(np.dot(face, eigenfaces[:num_eigen, :].T), eigenfaces[:num_eigen, :])
	reconst += mean
	return reconst


def compute_mse(reconst_face, ori_face):
	reconst_face = reconst_face.astype(np.uint8)
	return np.mean(np.power(reconst_face - ori_face, 2))


def project_to_eigen(x, eigens, num_eigen=None):
	if num_eigen is None:
		num_eigen = eigens.shape[0]

	return np.dot(x, eigens[:num_eigen, :].T)


def main():
	args = parse_input()

	if not os.path.isdir(args.faces_dir):
		print("Diretory %s doesnot exist" % args.faces_dir)
		exit(0)

	if args.face_npz is not None:
		if os.path.isfile(args.face_npz):
			all_faces = np.load(args.face_npz)
			train_faces, train_person, test_faces, test_person, img_shape = \
				all_faces['arr_0'], all_faces['arr_1'], all_faces['arr_2'], all_faces['arr_3'], all_faces['arr_4']
		else:
			train_faces, train_person, test_faces, test_person,  img_shape = read_all_imgs(path=args.faces_dir, save_path=args.face_npz)
	else:
		train_faces, train_person, test_faces, test_person, img_shape = read_all_imgs(path=args.faces_dir)

	train_faces = train_faces.astype(dtype=np.float32)
	test_faces = test_faces.astype(dtype=np.float32)

	# Create folder if not exists
	os.makedirs(Q1_DIR, exist_ok=True)
	os.makedirs(Q2_DIR, exist_ok=True)

	# Problem 1.1 -- Draw mean face
	mean_face = np.mean(train_faces, axis=0)
	plot_face(mean_face, os.path.join(Q1_DIR, "mean_face.png"), img_shape)

	train_faces -= mean_face
	test_faces -= mean_face

	# Problem 1.2 -- Draw top 3 eigenfaces
	eigenfaces, eigenvalues= pca(train_faces)
	plot_eigens(eigenfaces[:3], img_shape)

	# Problem 2.1 -- Reconstruct p1_f1 with diff eigenfces
	for num_eigenface in [3, 50, 100, 239]:
		p1f1_path = os.path.join(args.faces_dir, "1_1.png")
		ori_face = io.imread(p1f1_path)

		ori_face = ori_face.astype(dtype=np.float32)
		ori_face = ori_face.flatten()
		face = ori_face - mean_face

		reconst_face = reconstruct_faces(face, eigenfaces, mean_face, num_eigen=num_eigenface)
		# Problem 2.2 -- Compute MSE to original face
		mse = compute_mse(reconst_face, ori_face)
		print("Reconstruction using %d eigenfaces... MSE: %f" % (num_eigenface, mse))
		reconst_path = os.path.join(Q2_DIR, "%d_reconst.png" % num_eigenface)
		plot_face(reconst_face, reconst_path, img_shape)

	# Problem 3.1 -- Cross validation to find parameter
	from sklearn.model_selection import StratifiedKFold
	from sklearn.neighbors import KNeighborsClassifier

	n_split = 3
	skf = StratifiedKFold(n_splits=n_split)
	skf.get_n_splits(train_faces, train_person)

	accus = {}
	for train_index, val_index in skf.split(train_faces, train_person):
		train_X, val_X = train_faces[train_index], train_faces[val_index]
		train_y, val_y = train_person[train_index], train_person[val_index]

		eigenfaces, _ = pca(train_X)
		for num_eigen in [3, 50, 159]:
			train_projected_X = project_to_eigen(train_X, eigenfaces, num_eigen=num_eigen)
			val_projected_X = project_to_eigen(val_X, eigenfaces, num_eigen=num_eigen)

			for k in [1, 3, 5]:
				neigh = KNeighborsClassifier(n_neighbors=k)
				neigh.fit(train_projected_X, train_y)
				val_y_hat = neigh.predict(val_projected_X)
				accu = np.mean(val_y_hat == val_y)

				dict_idx = "%d_%d" % (num_eigen, k)
				if accus.get(dict_idx, None) is None:
					accus[dict_idx] = accu
				else:
					accus[dict_idx] += accu

	# Problem 3.2 -- Use best parameters on test set
	best_accu, best_num_eigen, best_k = 0., None, None
	for key in accus.keys():
		accus[key] /= n_split
		if accus[key] > best_accu:
			num_eigen, k = key.split('_')
			best_num_eigen, best_k = int(num_eigen), int(k)
			best_accu = accus[key]
		print("%s Accuracy: %f" % (key, accus[key]))

	eigenfacs, _ = pca(train_faces)
	train_projected_X = project_to_eigen(train_faces, eigenfaces, num_eigen=best_num_eigen)
	test_projected_X = project_to_eigen(test_faces, eigenfaces, num_eigen=best_num_eigen)

	neigh = KNeighborsClassifier(n_neighbors=best_k)
	neigh.fit(train_projected_X, train_person)
	test_y_hat = neigh.predict(test_projected_X)
	accu = np.mean(test_y_hat == test_person)
	print("Testing accuracy: %f on numeigen=%d, neigh=%d" % (accu, best_num_eigen, best_k))


if __name__ == '__main__':
	main()

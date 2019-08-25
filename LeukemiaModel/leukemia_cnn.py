def ambil_model():
	from keras.models import load_model
	model = load_model('my_model_sel_darah_putih.h5')
	return model
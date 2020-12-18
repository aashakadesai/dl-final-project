import numpy as np
import vggish_input
import csv
from pydub import AudioSegment

class_names = ['Finger_snapping', 'Glockenspiel','Tambourine','Cowbell','Gunshot_or_gunfire','Burping_or_eructation','Bass_drum','Oboe','Double_bass','Scissors','Telephone','Snare_drum']

fnames = []
labels = []

with open('train_data_list.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')

	for row in csv_reader:
		if row[1] in class_names:
			fnames.append(row[0])
			labels.append(row[1])

count = 62237
ms_names = []
ms_labels = []
ms_count = 0
mset_count = 123

#amplitude augmentation
for i in range(len(fnames)):
	f = 'audio_train/' + fnames[i]

	sound = AudioSegment.from_wav(f)
	avg_dbfs = sound.dBFS
	gain = avg_dbfs * 0.30
	louder = sound + gain
	softer = sound - gain

	louder.export('audio_train/louder' + fnames[i], format='wav')
	softer.export('audio_train/softer' + fnames[i], format='wav')

for i in range(len(fnames)):
	f = 'audio_train/louder' + fnames[i]
	label = labels[i]

	examples = vggish_input.wavfile_to_examples(f)
	if ms_count == 0:
		ms_data = examples
	else:
		#print("here")
		ms_data = np.concatenate((ms_data, examples))
	ms_count = ms_count + examples.shape[0]

	if ms_count > 500:
		name = 'train_combined/MSet' + str(mset_count) + '.npy'
		np.save(name, ms_data)
		print('SAVED')
		ms_count = 0
		mset_count = mset_count + 1

	for mel_spec in examples:
		name = 'MS' + str(count)
		ms_names.append(name)
		ms_labels.append(label)
		count = count + 1

for i in range(len(fnames)):
	f = 'audio_train/softer' + fnames[i]
	label = labels[i]

	examples = vggish_input.wavfile_to_examples(f)
	if ms_count == 0:
		ms_data = examples
	else:
		#print("here")
		ms_data = np.concatenate((ms_data, examples))
	ms_count = ms_count + examples.shape[0]

	if ms_count > 500:
		name = 'train_combined/MSet' + str(mset_count) + '.npy'
		np.save(name, ms_data)
		print('SAVED')
		ms_count = 0
		mset_count = mset_count + 1

	for mel_spec in examples:
		name = 'MS' + str(count)
		ms_names.append(name)
		ms_labels.append(label)
		count = count + 1

with open('train_amplified_data_ms_list.csv', mode = 'w') as ms_file:
	ms_writer = csv.writer(ms_file, delimiter=',')
	for i in range(len(ms_names)):
		ms_writer.writerow([ms_names[i], ms_labels[i]])


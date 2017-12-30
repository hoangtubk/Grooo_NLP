import io

class Convert():
	def __init__(self):
		self.dict_char_to_id = {}
		self.dict_accent_to_id = {}
		self.dict_id_to_char = {}
	def create_dict(self):
		# khởi tạo 2 dict lưu kí tự không dấu và các dấu câu
		id = 1
		for i in range(97, 123):
			self.dict_char_to_id[chr(i)] = id
			id += 1
		self.dict_char_to_id['<u>'] = 0
		self.dict_char_to_id[' '] = len(self.dict_char_to_id)
		list_accent = u'àáảãạăắằẵặẳâầấậẫẩđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
		print(len(list_accent))
		id = 1
		for i in list_accent:
			self.dict_accent_to_id[i] = id
			# chuyen doi tu co dau thanh khong dau
			if id < 18:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'a']
			if id == 18:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'd']
			if 18 < id < 30:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'e']
			if 29 < id < 35:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'i']
			if 34 < id < 52:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'o']
			if 51 < id < 63:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'u']
			if 62 < id < 68:
				self.dict_char_to_id[id] = self.dict_char_to_id[u'y']
			id +=1
		self.dict_accent_to_id['<u>'] = 0
		for key in self.dict_char_to_id.keys():
			self.dict_id_to_char[self.dict_char_to_id[key]] = key

	def convert_string_to_id(self, file_name):	
		string_without_accent = []
		accent = []
		with io.open(file_name,'r', encoding = 'utf8') as f:
			data = f.readlines()
		for i in range(0, len(data)):
			data[i] = data[i].split('\t')[0]
		data = [x.strip() for x in data] 
		for line in data:
			tmp1 = []
			tmp2 = []
			line = line.lower()
			for char in line:
				if char in self.dict_char_to_id:
					tmp1.append(self.dict_char_to_id[char])
				else:
					if char in self.dict_accent_to_id:
						tmp1.append(self.dict_char_to_id[self.dict_accent_to_id[char]])	
					else:	
						tmp1.append(28)
				if char in self.dict_accent_to_id:
					tmp2.append(self.dict_accent_to_id[char])
				else:
					tmp2.append(68)
			string_without_accent.append(tmp1)
			accent.append(tmp2)
		return (data, string_without_accent, accent)

	def convert_id_to_string(self, file_name):
		list_string = []
		with io.open(file_name,'r', encoding = 'utf8') as f:
			data = f.readlines()
		data = [x.strip() for x in data] 
		for line in data:
			tmp = []
			line = line.split()
			for id in line:
				id = int(id)
				tmp.append(self.dict_id_to_char[id])
			tmp = ''.join(tmp)
			list_string.append(tmp)
		return list_string


#test
if __name__ == '__main__':
	test = Convert()
	test.create_dict()
	file_name = 'beck_clean_all.txt'
	string_fn = '../data/' + file_name #data.raw
	(data, result1, result2) = test.convert_string_to_id(string_fn)
	# id_fn = './id.txt'
	# list_string = test.convert_id_to_string(id_fn)

	#save file input & target
	inp = open('../output/input_' + file_name, 'w')
	tar = open('../output/target_' + file_name, 'w')
	max_len = 0
	index_max = 0
	for i in range(len(data)):
		#get max_dim
		x = (str)(result1[i]).replace('[', '').replace(']', '').replace(', ', ' ')
		if len(x.split(' ')) > max_len:
			max_len = len(x.split(' '))
			index_max = i
		#
		inp.write((str)(result1[i]).replace('[', '').replace(']', '').replace(', ', ' ') + '\n')
		tar.write((str)(result2[i]).replace('[', '').replace(']', '').replace(', ', ' ') + '\n')
		print (data[i], '\n' ,result1[i], '\n', result2[i])
		print ("="*35)
	# print('max')
	print('max_dim')
	print (max_len)
	print('max_dim_index')
	print (index_max)
	inp.close()
	tar.close()
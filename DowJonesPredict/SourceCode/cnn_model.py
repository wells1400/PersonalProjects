# CNN建模处理
class CnnModel:
    def __init__(self,params):
        self.batch_size = params['batch_size']
        self.n_filter = params['n_filter']
        self.filter_length = params['filter_length']
        self.epochs = params['epochs']
        self.n_pool = params['n_pool']
        self.input_shape = params['input_shape']
        self.lr = params['lr']
        
        self.cnn_model = self.build_model()
    # 建立模型
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(self.n_filter,(self.filter_length,self.filter_length),
                        input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(self.n_filter,(self.filter_length,self.filter_length)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(self.n_pool, self.n_pool)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        # 后面接上一个ANN
        model.add(Dense(128))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        # compile模型
        adam_optimizer = Adam(lr=self.lr)
        model.compile(loss='binary_crossentropy',optimizer=adam_optimizer, metrics=['accuracy'])
        return model
    
    # 训练cnn模型
    def train_cnn_model(self, x_train, y_train, epochs=10,batch_size=4):
        self.cnn_model.fit(x_train,y_train,validation_split=0.33, epochs=epochs,batch_size=batch_size)
        
    # 衡量模型效果
    def evaluate_model(self,  x_test, y_test):
        score = self.cnn_model.evaluate(x_test, y_test)
        return score
    
    def run_engine(self, news_matrix, price_flag):
        selected_matrix = news_matrix
        selected_flags = price_flag
        xtrain,xtest,ytrain,ytest = train_test_split(selected_matrix, selected_flags,test_size=0.3)
        print(xtrain.shape)
        xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2],1)
        xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2],1)
        # 训练cnn模型
        self.train_cnn_model(xtrain, ytrain,epochs=self.epochs, batch_size=self.batch_size)
        # 衡量cnn模型效果
        score = self.evaluate_model(xtest, ytest)
        return score
OBJS = ./src/Main.cpp ./src/Matrix.cpp ./src/NeuralNetwork.cpp ./src/Gradient.cpp

CC = g++

OBJ_NAME = NNFS

all: $(OBJS)
	$(CC) $(OBJS) -o $(OBJ_NAME)

clean:
	rm -r $(OBJ_NAME).exe
CC=g++ -std=c++11

operation.o: operation.cpp
	$(CC) -c -o $@ $^

graph.o: graph.cpp
	$(CC) -c -o $@ $^

demo: demo.cpp graph.o operation.o
	$(CC) -o $@ $^

clean:
	rm -f demo graph.o operation.o
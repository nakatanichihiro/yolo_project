package socket;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;


/**
 * ソケット通信(サーバー側)
 */
class prepare_socket {

	void runSocket() {

	ServerSocket sSocket = null;
	Socket socket = null;
	BufferedReader reader = null;
	PrintWriter writer = null;

	try{
		
		//IPアドレスとポート番号を指定してサーバー側のソケットを作成
		sSocket = new ServerSocket();
		sSocket.bind(new InetSocketAddress
				("127.0.0.1",90));

		System.out.println("クライアントからの入力待ち状態");
		
//		get socket input for second seconds
		while(true) {
		socket = sSocket.accept();

//		read socket input
		reader = new BufferedReader(
				new InputStreamReader
				(socket.getInputStream()));


		String line = null;
  	
        	line = reader.readLine();
            System.out.println("クライアントで入力された文字＝" + line);
            
//            setting interval count to do yolo jud
            Thread.sleep(2000);
        }
	}catch(Exception e){
		e.printStackTrace();
	}finally{
		try{
			if (reader!=null){
				reader.close();
			}
			if (writer!=null){
				writer.close();
			}
			if (socket!=null){
				socket.close();
			}
			if (sSocket!=null){
				sSocket.close();
			}
            System.out.println("サーバー側終了です");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	}
}

public class show_socket {
	public static void main(String[] args) {
		prepare_socket s1 = new prepare_socket();
		s1.runSocket();
	}
}
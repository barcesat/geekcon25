#include <Arduino.h>
#include <Servo.h>


Servo myServo;

void setup() {
  // put your setup code here, to run once:
  // int result = myFunction(2, 3);
  myServo.attach(9); // Attach to pin 18 (change if needed)
  Serial.begin(9600);
  // Serial.println(result);
}

void loop() {
  // put your main code here, to run repeatedly:
  // Serial.println("Hello, World!");
  // delay(1000);
  // Serial.println("Looping...");
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    if (input == "SMILE") {
      Serial.println("slap");
      
      for (int i = 0; i < 2; ++i) {
        myServo.write(180);
        delay(500);
        myServo.write(0);
        delay(500);
      }
      myServo.detach();
    }

    // int incomingByte = Serial.read();
    // Serial.print("I received: ");
    // Serial.println(incomingByte, DEC);
  } 
}

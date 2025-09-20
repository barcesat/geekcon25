#include <Arduino.h>
#include <Servo.h>

#define ontime 500
#define offtime 500


Servo waterServo;
Servo fingerServo;

void setup() {
  pinMode(8, OUTPUT);
  waterServo.attach(9); // Attach to pin 18 (change if needed)
  fingerServo.attach(11);
  Serial.begin(9600);
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
        waterServo.write(180);
        fingerServo.write(180);
        digitalWrite(8 , HIGH);
        delay(ontime);
        digitalWrite(8, LOW);
        waterServo.write(0);
        fingerServo.write(0);
        delay(offtime);
    }

    // int incomingByte = Serial.read();
    // Serial.print("I received: ");
    // Serial.println(incomingByte, DEC);
  } 
}

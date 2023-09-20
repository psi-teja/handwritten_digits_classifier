import { useState, useEffect, SetStateAction, useRef } from "react";
import DrawingCanvas from "../components/DarwingCanvas";
import SelectModel from "../components/SelectModel";
import Prediction from "../components/Prediction";
import ImagesComponent from "../components/Images";

import styles from "./index.module.css";

export default function Home() {
  const [digit, setDigit] = useState("?");

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [modelID, setModelID] = useState(2);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    // Set initial drawing properties
    context.lineWidth = 20;
    context.strokeStyle = "black";
    context.lineCap = "round";

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function draw(e) {
      if (!isDrawing) return; // Stop if not drawing

      context.beginPath();
      context.moveTo(lastX, lastY);
      context.lineTo(
        e.clientX - canvas.offsetLeft,
        e.clientY - canvas.offsetTop
      );
      context.stroke();

      [lastX, lastY] = [
        e.clientX - canvas.offsetLeft,
        e.clientY - canvas.offsetTop,
      ];
    }

    canvas.addEventListener("mousedown", (e) => {
      isDrawing = true;
      [lastX, lastY] = [
        e.clientX - canvas.offsetLeft,
        e.clientY - canvas.offsetTop,
      ];
    });

    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", () => (isDrawing = false));
    canvas.addEventListener("mouseout", () => (isDrawing = false));
  }, []);

  const changeModel = (id: SetStateAction<number>) => {
    // Send a POST request to your Flask backend to change the model
    fetch(`http://localhost:3001/api/changemodel/${id}`, {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Model ID sent to backend", data); // Log the response message from the backend
        setModelID(id); // Set the active button based on the provided id
        setDigit("?");
      })
      .catch((error) => console.error("Error: error changing model", error));
  };

  const handleClearClick = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
  };

  const handleSubmitClick = () => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      // Get the pixel data from the entire canvas in grayscale color space
      const imageData = context.getImageData(
        0,
        0,
        canvas.width,
        canvas.height,
        { colorSpace: "srgb" }
      ).data;

      // Convert the pixel data to an array of pixel values
      const pixelValues = Array.from(imageData);

      const backendURL = "http://localhost:3001/api/upload"; // Replace with your backend URL

      // Send the pixel values to the server here
      fetch(backendURL, {
        method: "POST",
        body: JSON.stringify({ image: pixelValues }),
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          // Handle the response from the backend
          console.log("Pixel values uploaded successfully", data);
          setDigit(data.digit);
        })
        .catch((error) => {
          // Handle errors
          console.error("Error uploading pixel values", error);
        });
    }
  };

  return (
    <div>
      <div className={styles.container}>
        <div className={`${styles.component}`} style={{ flexBasis: "30%" }}>
          <DrawingCanvas
            canvasRef={canvasRef}
            handleClearClick={handleClearClick}
            handleSubmitClick={handleSubmitClick}
          />
        </div>
        <div className={`${styles.component}`} style={{ flexBasis: "40%" }}>
          <SelectModel modelID={modelID} changeModel={changeModel} />
        </div>
        <div className={`${styles.component}`} style={{ flexBasis: "30%" }}>
          <Prediction digit={digit} />
        </div>
      </div>
      <div className={styles.images}>
        <ImagesComponent modelID={modelID} />
      </div>
    </div>
  );
}

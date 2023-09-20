import Image from "next/image";
import styles from "./Images.module.css";

export default function ImagesComponent({ modelID }) {
  // Construct the image source URL based on the modelID
  const imageUrl = `/${modelID}.png`; // Adjust the path and file extension as needed

  return (
    <div className={styles.imageContainer}>
      <Image
        src={imageUrl}
        alt={`Image for Model ${modelID}`}
        width={700}
        height={300}
      />
    </div>
  );
}

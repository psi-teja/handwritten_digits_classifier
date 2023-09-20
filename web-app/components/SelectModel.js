import Button from "./Button.js";
import styles from "./SelectModel.module.css";

export default function SelectModel({ modelID, changeModel }) {
  // Destructure props here

  return (
    <div className={styles.container}>
      <h1>Models</h1>
      <Button
        id={1}
        text="MLPs using Numpy"
        handleClick={() => changeModel(1)}
        isActive={modelID === 1}
      />
      <Button
        id={2}
        text="CNN using TensorFlow"
        handleClick={() => changeModel(2)}
        isActive={modelID === 2}
      />
      <Button
        id={3}
        text="CNN using pyTorch"
        handleClick={() => changeModel(3)}
        isActive={modelID === 3}
      />
    </div>
  );
}

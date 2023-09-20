import styles from "./Prediction.module.css";

export default function Prediction({ digit }) {
  return (
    <div className={styles.container}>
      <h1>Prediction</h1>
      <h1 className={styles.digit}>{digit}</h1>
    </div>
  );
}

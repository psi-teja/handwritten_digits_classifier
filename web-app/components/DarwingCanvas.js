import styles from './DarwingCanvas.module.css'

export default function DrawingCanvas({canvasRef, handleClearClick, handleSubmitClick}) {

  return (
    <div className={styles.container}>
      <h1>Drawing Canvas</h1>
      <canvas className={styles.canvas} ref={canvasRef} width={280} height={280}></canvas>
      <div className={styles['button-container']}>
        <button onClick={handleClearClick}>clear</button>
        <button onClick={handleSubmitClick}>submit</button>
      </div>
    </div>
  );

}

import React, { useState } from 'react';
import styles from './Button.module.css';

function Button({text, id, handleClick, isActive}) {

  return (
    <div>
      <button
        className={`${styles.button} ${isActive ? styles.active : ''}`}
        onClick={() => handleClick(id)}
      >
        {text}
      </button>
    </div>
  );
}

export default Button;

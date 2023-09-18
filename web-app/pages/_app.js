import { useEffect } from 'react';

function MyApp({ Component, pageProps }) {
  useEffect(() => {
    document.title = 'Hand Written Digits Classifier';
  }, []);

  return <Component {...pageProps} />;
}

export default MyApp;

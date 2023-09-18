/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
      remotePatterns: [
        {
          protocol: 'https',
          hostname: 'www.researchgate.net',
          pathname: '/publication/**',
        },
      ],
    },
  }

module.exports = nextConfig

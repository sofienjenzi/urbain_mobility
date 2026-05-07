export default function Logo({ size = 'md' }) {
  const sizes = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-28 h-28',
    '2xl': 'w-40 h-40',
  };

  return (
    <div className={`${sizes[size]} relative`}>
      <svg
        viewBox="0 0 400 320"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full drop-shadow-lg"
      >
        {/* Background */}
        <rect width="400" height="320" fill="none" rx="20" />

        {/* Left pillar */}
        <rect x="60" y="80" width="60" height="140" fill="#0d9488" rx="8" />

        {/* Right pillar */}
        <rect x="280" y="80" width="60" height="140" fill="#0d9488" rx="8" />

        {/* Bottom connector */}
        <rect x="60" y="210" width="280" height="30" fill="#0d9488" rx="6" />

        {/* Chart line - animated path */}
        <path
          d="M 90 160 Q 130 120 170 140 T 250 100 L 310 140"
          stroke="#f59e0b"
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          className="animate-pulse"
        />

        {/* Data points on line */}
        <circle cx="90" cy="160" r="6" fill="#f59e0b" className="animate-bounce" />
        <circle cx="170" cy="140" r="6" fill="#f59e0b" />
        <circle cx="250" cy="100" r="6" fill="#f59e0b" />
        <circle cx="310" cy="140" r="6" fill="#f59e0b" className="animate-pulse" />
      </svg>
    </div>
  );
}

import AdminDashboard from './dashboards/AdminDashboard';
import MinisterDashboard from './dashboards/MinisterDashboard';
import AirParifDashboard from './dashboards/AirParifDashboard';
import CitizenDashboard from './dashboards/CitizenDashboard';
import TransportDashboard from './dashboards/TransportDashboard';

export default function Dashboard({ user }) {
  const renderDashboard = () => {
    switch (user.role) {
      case 'admin':
        return <AdminDashboard user={user} />;
      case 'minister':
        return <MinisterDashboard user={user} />;
      case 'air_parif':
        return <AirParifDashboard user={user} />;
      case 'citizen':
        return <CitizenDashboard user={user} />;
      case 'transport':
        return <TransportDashboard user={user} />;
      default:
        return <div>Rôle inconnu</div>;
    }
  };

  return <div>{renderDashboard()}</div>;
}

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { TrendingUp, TrendingDown, DollarSign, MousePointer, Eye, ArrowUpRight, Target, Users } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, Legend } from 'recharts'

interface Stats {
  totalSpend: number
  impressions: number
  clicks: number
  conversions: number
  ctr: number
  roas: number
  budgetUsed: number
}

const roiData = [
  { date: 'Mar 9', roas: 2.8, predicted: 2.9 },
  { date: 'Mar 10', roas: 3.1, predicted: 3.0 },
  { date: 'Mar 11', roas: 2.9, predicted: 3.1 },
  { date: 'Mar 12', roas: 3.4, predicted: 3.2 },
  { date: 'Mar 13', roas: 3.2, predicted: 3.3 },
  { date: 'Mar 14', roas: 3.5, predicted: 3.4 },
  { date: 'Mar 15', roas: 3.8, predicted: 3.6 },
]

// PPO Bidding Dynamics - 展示 AI 实时出价调整
const ppoBidData = [
  { time: '00:00', delta: 0.1 },
  { time: '04:00', delta: -0.05 },
  { time: '08:00', delta: 0.15 },
  { time: '12:00', delta: 0.25 },
  { time: '16:00', delta: 0.2 },
  { time: '20:00', delta: 0.1 },
  { time: '24:00', delta: 0.05 },
]

const funnelData = [
  { name: 'Impressions', value: 2456789, fill: '#3B82F6' },
  { name: 'Clicks', value: 45678, fill: '#10B981' },
  { name: 'Conversions', value: 2345, fill: '#F59E0B' },
]

const budgetData = [
  { name: 'Spent', value: 4500 },
  { name: 'Remaining', value: 5500 },
]

export default function Dashboard() {
  const [stats] = useState<Stats>({
    totalSpend: 12450,
    impressions: 2456789,
    clicks: 45678,
    conversions: 2345,
    ctr: 1.86,
    roas: 3.2,
    budgetUsed: 45,
  })

  const metricCards = [
    { name: 'Total Spend', value: `$${stats.totalSpend.toLocaleString()}`, change: '+12%', trend: 'up', icon: DollarSign },
    { name: 'Impressions', value: (stats.impressions / 1000).toFixed(1) + 'K', change: '+8%', trend: 'up', icon: Eye },
    { name: 'Clicks', value: (stats.clicks / 1000).toFixed(1) + 'K', change: '+15%', trend: 'up', icon: MousePointer },
    { name: 'CTR', value: stats.ctr + '%', change: '+0.2%', trend: 'up', icon: Target },
    { name: 'ROAS', value: stats.roas + 'x', change: '+0.5x', trend: 'up', icon: TrendingUp },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500">Overview of your advertising performance</p>
        </div>
        <Link
          to="/campaigns/new"
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
        >
          <ArrowUpRight className="w-4 h-4 mr-2" />
          New Campaign
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        {metricCards.map((metric) => (
          <div key={metric.name} className="overflow-hidden bg-white rounded-lg shadow">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <metric.icon className="w-6 h-6 text-gray-400" />
                </div>
                <div className="flex-1 ml-3">
                  <dt className="text-sm font-medium text-gray-500 truncate">{metric.name}</dt>
                  <dd className="flex items-baseline">
                    <div className="text-2xl font-semibold text-gray-900">{metric.value}</div>
                    <div className={`ml-2 text-sm font-medium ${metric.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                      {metric.change}
                    </div>
                  </dd>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row 1: ROI & Funnel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROI Trend Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">ROI Trend (Actual vs ML Prediction)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={roiData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                <YAxis stroke="#6B7280" fontSize={12} domain={[0, 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                />
                <Legend />
                <Line type="monotone" dataKey="roas" name="Actual ROAS" stroke="#10B981" strokeWidth={2} dot={{ fill: '#10B981', r: 4 }} />
                <Line type="monotone" dataKey="predicted" name="ML Predicted" stroke="#8B5CF6" strokeWidth={2} strokeDasharray="5 5" dot={{ fill: '#8B5CF6', r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Conversion Funnel */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Conversion Funnel</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={funnelData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis type="number" stroke="#6B7280" fontSize={12} />
                <YAxis dataKey="name" type="category" stroke="#6B7280" fontSize={12} width="auto" minWidth={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                  formatter={(value: number) => [value.toLocaleString(), 'Count']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {funnelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* PPO Bidding Dynamics */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          PPO Bidding Dynamics 
          <span className="ml-2 text-xs font-normal text-gray-500">(AI Strategy Accountability)</span>
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ppoBidData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={12} domain={[-0.5, 0.5]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Bid Adjustment (δ)']}
              />
              <Line type="monotone" dataKey="delta" name="PPO δ" stroke="#F59E0B" strokeWidth={2} dot={{ fill: '#F59E0B', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-gray-500">
          Real-time bid adjustment coefficient (δ) from PPO Agent. Positive values indicate aggressive bidding.
        </p>
      </div>

      {/* Charts Row 2: Budget & Channel Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Budget Gauge */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Budget Utilization</h3>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={budgetData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                >
                  <Cell fill="#3B82F6" />
                  <Cell fill="#E5E7EB" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900">{stats.budgetUsed}%</div>
              <div className="text-sm text-gray-500">of $10,000</div>
            </div>
          </div>
        </div>

        {/* Channel Distribution */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Spend by Channel</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Google', value: 45, color: '#4285F4' },
                    { name: 'Facebook', value: 25, color: '#1877F2' },
                    { name: 'LinkedIn', value: 15, color: '#0A66C2' },
                    { name: 'Other', value: 15, color: '#9CA3AF' },
                  ]}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {[
                    { name: 'Google', value: 45, color: '#4285F4' },
                    { name: 'Facebook', value: 25, color: '#1877F2' },
                    { name: 'LinkedIn', value: 15, color: '#0A66C2' },
                    { name: 'Other', value: 15, color: '#9CA3AF' },
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Campaigns */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Active Campaigns</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {[
            { name: 'Summer Sale 2024', status: 'Active', spend: 4500, ctr: 2.1 },
            { name: 'Brand Awareness', status: 'Active', spend: 3200, ctr: 1.8 },
            { name: 'Product Launch', status: 'Paused', spend: 2100, ctr: 1.5 },
          ].map((campaign) => (
            <div key={campaign.name} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50">
              <div>
                <p className="text-sm font-medium text-gray-900">{campaign.name}</p>
                <p className="text-sm text-gray-500">{campaign.status}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">${campaign.spend.toLocaleString()}</p>
                <p className="text-sm text-gray-500">CTR: {campaign.ctr}%</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

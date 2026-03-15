import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'

const dailyData = [
  { date: 'Mar 1', impressions: 45000, clicks: 890, spend: 234 },
  { date: 'Mar 2', impressions: 52000, clicks: 1020, spend: 280 },
  { date: 'Mar 3', impressions: 49000, clicks: 950, spend: 265 },
  { date: 'Mar 4', impressions: 61000, clicks: 1250, spend: 340 },
  { date: 'Mar 5', impressions: 55000, clicks: 1080, spend: 295 },
  { date: 'Mar 6', impressions: 67000, clicks: 1340, spend: 380 },
  { date: 'Mar 7', impressions: 72000, clicks: 1480, spend: 420 },
]

const channelData = [
  { name: 'Google Ads', value: 45, color: '#4285F4' },
  { name: 'Facebook', value: 25, color: '#1877F2' },
  { name: 'LinkedIn', value: 15, color: '#0A66C2' },
  { name: 'Other', value: 15, color: '#9CA3AF' },
]

export default function Analytics() {
  const [dateRange, setDateRange] = useState('7d')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="mt-1 text-sm text-gray-500">Deep dive into your campaign performance</p>
        </div>
        <select
          value={dateRange}
          onChange={(e) => setDateRange(e.target.value)}
          className="block w-40 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
        >
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
        </select>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Impressions & Clicks Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Impressions & Clicks</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                <YAxis yAxisId="left" stroke="#6B7280" fontSize={12} />
                <YAxis yAxisId="right" orientation="right" stroke="#6B7280" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                />
                <Line yAxisId="left" type="monotone" dataKey="impressions" stroke="#3B82F6" strokeWidth={2} dot={false} name="Impressions" />
                <Line yAxisId="right" type="monotone" dataKey="clicks" stroke="#10B981" strokeWidth={2} dot={false} name="Clicks" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spend Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Daily Spend</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                <YAxis stroke="#6B7280" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                  formatter={(value: number) => [`$${value}`, 'Spend']}
                />
                <Bar dataKey="spend" fill="#8B5CF6" radius={[4, 4, 0, 0]} name="Spend" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Channel Distribution */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Spend by Channel</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={channelData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {channelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {channelData.map((channel) => (
              <div key={channel.name} className="flex items-center">
                <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: channel.color }} />
                <span className="text-sm text-gray-600">{channel.name}</span>
                <span className="ml-auto text-sm font-medium text-gray-900">{channel.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* CTR Trend */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">CTR Trend</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                <YAxis stroke="#6B7280" fontSize={12} domain={[0, 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px' }}
                  formatter={(value: number) => [`${((value / 1000) * 100).toFixed(2)}%`, 'CTR']}
                />
                <Line
                  type="monotone"
                  dataKey="clicks"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={{ fill: '#F59E0B', r: 4 }}
                  name="CTR"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
